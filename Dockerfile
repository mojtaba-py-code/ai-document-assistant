# syntax=docker/dockerfile:1.7
#
# AI Document Assistant (docassist) - production image.
#
#   docker build -t docassist:1.0.0 .
#   docker build -t docassist:1.0.0-ocr --build-arg WITH_OCR=true .
#
# * Multi-stage: uv, compilers, caches and the build context never reach the runtime image.
# * Dependencies are installed from a hash-pinned requirements file (uv.lock when committed,
#   otherwise resolved from pyproject.toml) with --require-hashes; dev tools are excluded.
# * Runs as uid/gid 10001 without a login shell; setuid/setgid bits are stripped.
# * Works with a read-only root filesystem: the only writable paths are the state volume
#   /var/lib/docassist and a tmpfs mounted at /tmp (compose.yaml / the Kubernetes manifests
#   provide both - run with `--read-only --tmpfs /tmp`).
# * Secure by default: DOCASSIST_ENVIRONMENT=production, which refuses to start with
#   development shortcuts. Secrets are never baked in; pass them as files and point the
#   DOCASSIST_*_FILE variables at them (Docker/Kubernetes secrets).
#
# Base images are pinned by digest (resolved 2026-09-30); Dependabot (docker ecosystem)
# proposes digest bumps weekly. To re-resolve by hand:
#   docker buildx imagetools inspect python:3.12-slim-bookworm --format '{{.Manifest.Digest}}'

# ------------------------------------------------------------------------------------------
# Build stage
# ------------------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 /uv /usr/local/bin/uv

# Optional extras: qdrant (vector store client) is always installed so the same image serves
# both retrieval back-ends; WITH_OCR=true adds the "ocr" extra here and tesseract below.
ARG WITH_OCR=false

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_PROGRESS=1

WORKDIR /src

RUN uv venv --python /usr/local/bin/python3.12 /opt/venv

# 1) Third-party dependencies (cached until pyproject.toml / uv.lock change).
#    `uv.lock*` copies the lock file when it exists and is a no-op otherwise.
COPY pyproject.toml uv.lock* README.md LICENSE ./
COPY scripts/export-requirements.sh ./scripts/export-requirements.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    extras="--extra qdrant"; \
    if [ "$WITH_OCR" = "true" ]; then extras="$extras --extra ocr"; fi; \
    sh scripts/export-requirements.sh --output /tmp/requirements.txt $extras; \
    uv pip install --python /opt/venv/bin/python --require-hashes --no-deps \
        --requirement /tmp/requirements.txt

# 2) The application wheel, installed without dependencies (they are all pinned above). The
#    wheel bundles the Alembic migrations (pyproject.toml force-include), so the build needs them.
COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    uv build --wheel --out-dir /tmp/dist; \
    uv pip install --python /opt/venv/bin/python --no-deps /tmp/dist/docassist-*.whl

# ------------------------------------------------------------------------------------------
# Runtime stage
# ------------------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime

ARG WITH_OCR=false
ARG VERSION=1.0.0
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="AI Document Assistant" \
      org.opencontainers.image.description="Secure multi-tenant document intelligence and RAG platform (API and worker)" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="Mojtaba Karimi" \
      org.opencontainers.image.authors="Mojtaba Karimi" \
      org.opencontainers.image.source="https://github.com/mojtaba-py-code/ai-document-assistant" \
      org.opencontainers.image.documentation="https://github.com/mojtaba-py-code/ai-document-assistant/tree/main/docs" \
      org.opencontainers.image.base.name="docker.io/library/python:3.12-slim-bookworm"

# `apt-get upgrade` applies the Debian security fixes published since the pinned base image
# was built (CI never restores this stage from its cache, so each build gets them). OCR
# (optional) is the only package installed on top of the base image. Then: a dedicated
# unprivileged user, the state directory, no pip in the runtime, no setuid/setgid binaries.
RUN set -eu; \
    apt-get update; \
    apt-get upgrade -y --no-install-recommends; \
    if [ "$WITH_OCR" = "true" ]; then \
        apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng; \
    fi; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd --system --gid 10001 docassist; \
    useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --no-create-home \
        --shell /usr/sbin/nologin docassist; \
    install -d -o 10001 -g 10001 -m 0750 /var/lib/docassist; \
    install -d -o root -g root -m 0755 /app; \
    ln -s /var/lib/docassist /app/var; \
    python -m pip uninstall --yes --quiet pip; \
    find / -xdev -type f -perm /6000 -exec chmod a-s {} + 2>/dev/null || true

COPY --from=builder /opt/venv /opt/venv
# `alembic -c /app/alembic.ini` (compose, the Kubernetes migration Job) runs from this
# root-owned, read-only copy; `docassist migrate` also finds the copy bundled in the wheel.
COPY alembic.ini /app/alembic.ini
COPY migrations /app/migrations

# /app/var -> /var/lib/docassist, so relative "var/..." paths (storage root default, the
# development e-mail outbox) always land on the state volume, never on the read-only image.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    LANG=C.UTF-8 \
    DOCASSIST_ENVIRONMENT=production \
    DOCASSIST_STORAGE__ROOT=/var/lib/docassist/storage \
    DOCASSIST_OBSERVABILITY__LOG_FORMAT=json

USER 10001:10001
WORKDIR /app
VOLUME ["/var/lib/docassist"]
EXPOSE 8000

# Liveness only (no dependencies): readiness is /health/ready. The worker has no HTTP port,
# so compose.yaml and the Kubernetes manifests override this check for the worker.
# 127.0.0.1 must stay in DOCASSIST_SECURITY__ALLOWED_HOSTS (TrustedHostMiddleware).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status == 200 else 1)"]

STOPSIGNAL SIGTERM
ENTRYPOINT ["docassist"]
# Inside a container the server must listen on all interfaces; publish the port only where
# intended (compose publishes it on 127.0.0.1).
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
