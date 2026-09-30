#!/usr/bin/env bash
# Disposable PostgreSQL 16 + pgvector for local development and the database test suite.
#
#   scripts/dev-db.sh up        start (or reuse) the container, wait until it accepts connections
#   scripts/dev-db.sh env       print the path of the env file to source for the tests
#   scripts/dev-db.sh psql      open psql as the superuser
#   scripts/dev-db.sh down      stop and remove the container (the data volume is kept)
#   scripts/dev-db.sh destroy   remove the container AND its data volume
#
# The server listens on 127.0.0.1 only (port DOCASSIST_DEV_DB_PORT, default 55432). A random
# superuser password is generated once into var/dev-db/password (mode 0600) and never printed;
# `up` writes var/dev-db/test.env with DOCASSIST_TEST_DATABASE_URL for the test suite:
#
#   scripts/dev-db.sh up && set -a && . var/dev-db/test.env && set +a
#   python -m pytest -m db
#
# The test fixtures create a throw-away database, the docassist_app/docassist_worker roles and
# run the real migrations themselves (tests/conftest.py).
#
# Environment: DOCKER (default docker), DOCASSIST_DEV_DB_PORT, DOCASSIST_DEV_DB_IMAGE.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="${root_dir}/var/dev-db"
container="docassist-dev-db"
volume="docassist-dev-db-data"
port="${DOCASSIST_DEV_DB_PORT:-55432}"
image="${DOCASSIST_DEV_DB_IMAGE:-pgvector/pgvector:0.8.6-pg16-bookworm}"
read -r -a docker <<<"${DOCKER:-docker}"

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; }

[[ "$port" =~ ^[0-9]{1,5}$ ]] || { echo "dev-db: DOCASSIST_DEV_DB_PORT must be a port number" >&2; exit 2; }

password_file="${state_dir}/password"
env_file="${state_dir}/test.env"

ensure_password() {
    umask 077
    mkdir -p -- "$state_dir"
    if [[ ! -s "$password_file" ]]; then
        openssl rand -hex 24 | tr -d '\r\n' >"$password_file"
    fi
    chmod 0600 -- "$password_file"
}

running() { [[ "$("${docker[@]}" inspect -f '{{.State.Running}}' "$container" 2>/dev/null)" == "true" ]]; }

cmd_up() {
    ensure_password
    if ! "${docker[@]}" inspect "$container" >/dev/null 2>&1; then
        # The password reaches the container as a read-only file mount, not as an environment
        # value visible in `docker inspect`.
        "${docker[@]}" run -d --name "$container" \
            --publish "127.0.0.1:${port}:5432" \
            --volume "${volume}:/var/lib/postgresql/data" \
            --mount "type=bind,source=${password_file},target=/run/secrets/postgres_password,readonly" \
            --env POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password \
            --env POSTGRES_INITDB_ARGS="--auth-host=scram-sha-256" \
            --security-opt no-new-privileges:true \
            "$image" >/dev/null
    elif ! running; then
        "${docker[@]}" start "$container" >/dev/null
    fi
    printf 'dev-db: waiting for PostgreSQL on 127.0.0.1:%s ' "$port" >&2
    for _ in $(seq 1 60); do
        if "${docker[@]}" exec "$container" pg_isready -q -h 127.0.0.1 -U postgres >/dev/null 2>&1; then
            echo "ready" >&2
            write_env
            return 0
        fi
        printf '.' >&2
        sleep 1
    done
    echo "timed out" >&2
    "${docker[@]}" logs --tail 50 "$container" >&2 || true
    return 1
}

write_env() {
    umask 077
    printf 'DOCASSIST_TEST_DATABASE_URL=postgresql://postgres:%s@127.0.0.1:%s/postgres\n' \
        "$(cat "$password_file")" "$port" >"$env_file"
    chmod 0600 -- "$env_file"
    echo "dev-db: test settings in ${env_file} (source it; it contains the password)" >&2
}

cmd_psql() {
    running || { echo "dev-db: not running (scripts/dev-db.sh up)" >&2; exit 1; }
    exec "${docker[@]}" exec -it -e PGPASSWORD_FILE=/run/secrets/postgres_password "$container" \
        sh -c 'PGPASSWORD="$(cat "$PGPASSWORD_FILE")" exec psql -h 127.0.0.1 -U postgres'
}

case "${1:-}" in
    up) cmd_up ;;
    env)
        [[ -s "$env_file" ]] || { echo "dev-db: run 'scripts/dev-db.sh up' first" >&2; exit 1; }
        echo "$env_file"
        ;;
    psql) cmd_psql ;;
    down) "${docker[@]}" rm -f "$container" >/dev/null && echo "dev-db: container removed (volume ${volume} kept)" >&2 ;;
    destroy)
        "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
        "${docker[@]}" volume rm "$volume" >/dev/null && echo "dev-db: container and volume removed" >&2
        ;;
    -h | --help | help) usage ;;
    *) usage >&2; exit 2 ;;
esac
