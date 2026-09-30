#!/usr/bin/env bash
# Restore a backup made by scripts/backup.sh into the docker-compose deployment.
#
#   scripts/restore.sh --drill BACKUP_DIR     restore drill (safe, run it regularly)
#   scripts/restore.sh --apply BACKUP_DIR     replace the live database and object store
#
# --drill  verifies SHA256SUMS, restores the dump into a throw-away database next to the
#          live one, checks the schema version and row counts, verifies the storage archive
#          can be read, prints a report and drops the throw-away database. The live data is
#          never touched.
# --apply  stops api and worker, restores the database (pg_restore --clean, one
#          transaction) and the object store, then starts api and worker again. It asks for
#          confirmation (type the database name) unless RESTORE_CONFIRM=docassist is set.
#
# Encrypted backups (*.age) need BACKUP_AGE_IDENTITY=<path to the age identity file>.
# The roles docassist_owner/app/worker must exist in the target cluster (they are created by
# deployment/postgres/initdb on a fresh volume) - the dump references them for ownership,
# grants and row-level-security policies.
#
# Environment: COMPOSE (default "docker compose"), BACKUP_AGE_IDENTITY, RESTORE_CONFIRM.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
db_name="docassist"
read -r -a compose <<<"${COMPOSE:-docker compose}"

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; }

mode=""
backup_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --drill | --apply)
            mode=${1#--}
            [[ $# -ge 2 ]] || { echo "restore: $1 needs a backup directory" >&2; exit 2; }
            backup_dir=$2
            shift 2
            ;;
        -h | --help) usage; exit 0 ;;
        *) echo "restore: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$mode" ]] || { usage >&2; exit 2; }
[[ -d "$backup_dir" ]] || { echo "restore: no such directory: $backup_dir" >&2; exit 2; }
backup_dir="$(cd -- "$backup_dir" && pwd)"
cd -- "$root_dir"

# --- 1. integrity ---------------------------------------------------------------------------
(
    cd -- "$backup_dir"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum --check --quiet SHA256SUMS
    else
        shasum -a 256 --check --quiet SHA256SUMS
    fi
) || { echo "restore: checksum verification FAILED - backup is corrupt or was modified" >&2; exit 1; }
echo "restore: checksums OK" >&2

# decrypt NAME: write the plaintext of NAME (or NAME.age) to stdout.
decrypt() {
    local base="${backup_dir}/$1"
    if [[ -f "${base}.age" ]]; then
        [[ -n "${BACKUP_AGE_IDENTITY:-}" ]] || { echo "restore: $1 is encrypted; set BACKUP_AGE_IDENTITY" >&2; return 1; }
        age --decrypt --identity "$BACKUP_AGE_IDENTITY" "${base}.age"
    elif [[ -f "$base" ]]; then
        cat -- "$base"
    else
        echo "restore: $1 not found in the backup" >&2
        return 1
    fi
}

has_storage() { [[ -f "${backup_dir}/storage.tar.gz" || -f "${backup_dir}/storage.tar.gz.age" ]]; }

# psql_super DB SQL...: run SQL as the superuser inside the postgres container (the password
# is read from the secret file there, never passed through this host).
psql_super() {
    local db=$1
    shift
    "${compose[@]}" exec -T postgres sh -ceu '
        PGPASSWORD="$(cat /run/secrets/postgres_superuser_password)"
        export PGPASSWORD
        db=$1
        shift
        exec psql --no-psqlrc --username=postgres --dbname="$db" -v ON_ERROR_STOP=1 -At "$@"
    ' psql "$db" "$@"
}

# pg_restore_into DB [OPTIONS...]: stream the dump from this host into pg_restore.
pg_restore_into() {
    local db=$1
    shift
    decrypt database.dump | "${compose[@]}" exec -T postgres sh -ceu '
        PGPASSWORD="$(cat /run/secrets/postgres_superuser_password)"
        export PGPASSWORD
        db=$1
        shift
        exec pg_restore --username=postgres --dbname="$db" --exit-on-error --no-password "$@"
    ' pg_restore "$db" "$@"
}

report_counts() {
    local db=$1
    psql_super "$db" \
        -c "SELECT 'schema version: ' || version_num FROM alembic_version" \
        -c "SELECT 'organizations: ' || count(*) FROM organizations" \
        -c "SELECT 'users: ' || count(*) FROM users" \
        -c "SELECT 'documents: ' || count(*) FROM documents" \
        -c "SELECT 'document chunks: ' || count(*) FROM document_chunks" \
        -c "SELECT 'audit events: ' || count(*) FROM audit_events" \
        -c "SELECT 'row-level security tables: ' || count(*) FROM pg_class WHERE relrowsecurity"
}

if [[ $mode == drill ]]; then
    drill_db="docassist_restore_drill_$(date -u +%Y%m%d%H%M%S)"
    cleanup() { psql_super postgres -c "DROP DATABASE IF EXISTS ${drill_db} WITH (FORCE)" >/dev/null 2>&1 || true; }
    trap cleanup EXIT

    echo "restore: drill database ${drill_db}" >&2
    psql_super postgres -c "CREATE DATABASE ${drill_db} OWNER docassist_owner TEMPLATE template0 ENCODING 'UTF8'" >/dev/null
    pg_restore_into "$drill_db" --single-transaction
    echo "restore: database restored; contents:" >&2
    report_counts "$drill_db"

    if has_storage; then
        entries="$(decrypt storage.tar.gz | tar -tzf - | wc -l | tr -d ' ')"
        echo "storage archive entries: ${entries}"
    fi
    echo "restore: DRILL PASSED (the throw-away database is dropped on exit)" >&2
    exit 0
fi

# --- apply ------------------------------------------------------------------------------------
if [[ "${RESTORE_CONFIRM:-}" != "$db_name" ]]; then
    printf 'This REPLACES the live database "%s" and the object store. Type the database name to continue: ' "$db_name" >&2
    read -r answer
    [[ "$answer" == "$db_name" ]] || { echo "restore: aborted" >&2; exit 1; }
fi

echo "restore: stopping api and worker" >&2
"${compose[@]}" stop api worker

pg_restore_into "$db_name" --clean --if-exists --single-transaction
echo "restore: database restored" >&2
report_counts "$db_name"

if has_storage; then
    decrypt storage.tar.gz | "${compose[@]}" run --rm -T --no-deps --entrypoint sh api -ceu \
        'cd /var/lib/docassist && rm -rf storage.restoring && mkdir storage.restoring && tar -xzf - -C storage.restoring && rm -rf storage && mv storage.restoring/storage storage && rmdir storage.restoring'
    echo "restore: object store restored" >&2
fi

echo "restore: starting api and worker" >&2
"${compose[@]}" up -d api worker
echo "restore: done - verify the audit chain: docker compose exec api docassist audit-verify" >&2
