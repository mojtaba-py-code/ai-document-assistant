# Developer shortcuts. Every target is a thin wrapper around uv, docker compose or a script in
# scripts/, so CI and humans run exactly the same commands (.github/workflows/ci.yml).
.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

UV ?= uv
COMPOSE ?= docker compose
IMAGE ?= docassist:1.0.0
SBOM ?= sbom-docassist.spdx.json

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

# --- local development --------------------------------------------------------------------
install: ## Install the project with dev tools and all extras (locked if uv.lock exists)
	@if [ -f uv.lock ]; then $(UV) sync --locked --all-extras; else $(UV) sync --all-extras; fi

env: ## Create a development .env with fresh random secrets (refuses to overwrite)
	$(UV) run docassist init-env

lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Format and auto-fix
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## mypy (strict)
	$(UV) run mypy

arch: ## Architecture contracts (import-linter)
	$(UV) run lint-imports

test: ## Test suite without a database (db tests are skipped)
	$(UV) run pytest -p no:cacheprovider

test-db: ## Full suite incl. PostgreSQL tests (needs DOCASSIST_TEST_DATABASE_URL, see scripts/dev-db.sh)
	@test -n "$${DOCASSIST_TEST_DATABASE_URL:-}" || { echo "DOCASSIST_TEST_DATABASE_URL is not set: scripts/dev-db.sh up && set -a && . var/dev-db/test.env && set +a" >&2; exit 2; }
	$(UV) run pytest -p no:cacheprovider --cov=docassist --cov-report=term-missing

security: ## Bandit + pip-audit on the hash-pinned runtime requirements
	$(UV) run bandit -c pyproject.toml -r src
	sh scripts/export-requirements.sh --output requirements.audit.txt --all-extras
	$(UV) run pip-audit --requirement requirements.audit.txt --require-hashes --disable-pip --progress-spinner off
	@rm -f requirements.audit.txt

check: lint typecheck arch test security ## Everything CI runs except the container job

migrate: ## Apply database migrations (schema owner DSN from .env)
	$(UV) run docassist migrate

run: ## API + web UI on http://127.0.0.1:8000
	$(UV) run docassist serve --host 127.0.0.1 --port 8000

worker: ## Background worker (jobs + maintenance)
	$(UV) run docassist worker

seed: ## Demo organisations, users and documents (passwords go to .demo-credentials, mode 0600)
	$(UV) run docassist seed-demo --password-file .demo-credentials

dev-db: ## Disposable PostgreSQL 16 + pgvector on 127.0.0.1 for tests
	bash scripts/dev-db.sh up

# --- containers ---------------------------------------------------------------------------
secrets: ## Generate deployment/secrets/* for docker compose (never overwrites)
	bash scripts/generate-secrets.sh

docker-build: ## Build the container image
	docker build --build-arg VCS_REF="$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" -t $(IMAGE) .

compose-up: secrets ## Start the default stack (offline AI) on http://127.0.0.1:8000
	$(COMPOSE) up -d --build

compose-down: ## Stop the stack (volumes are kept)
	$(COMPOSE) down

compose-logs: ## Follow the api and worker logs
	$(COMPOSE) logs -f api worker

backup: ## Back up the compose database and object store into backups/
	bash scripts/backup.sh

restore-drill: ## Restore the newest backup into a throw-away database and verify it
	@latest="$$(ls -1d backups/*/ 2>/dev/null | sort | tail -n 1)"; \
	test -n "$$latest" || { echo "no backups/ found - run make backup" >&2; exit 2; }; \
	bash scripts/restore.sh --drill "$$latest"

sbom: ## SPDX SBOM of the container image (needs syft)
	syft $(IMAGE) -o spdx-json=$(SBOM)

.PHONY: help install env lint format typecheck arch test test-db security check migrate run \
	worker seed dev-db secrets docker-build compose-up compose-down compose-logs backup \
	restore-drill sbom
