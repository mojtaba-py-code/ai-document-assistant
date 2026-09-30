#!/usr/bin/env bash
# Back up a docker-compose deployment of docassist:
#   * the database: pg_dump (custom format, includes ownership, RLS policies and grants),
#   * the object store: the encrypted blobs under /var/lib/docassist/storage,
#   * SHA256SUMS over both, so restore.sh can prove the files are intact.
#
#   scripts/backup.sh [--output-dir DIR] [--no-storage]
#
# Backups contain document text (chunks live in PostgreSQL in clear text for search), so:
#   * the backup directory is created 0700;
#   * set BACKUP_AGE_RECIPIENT to an age public key (https://age-encryption.org) to encrypt
#     every file before it touches the disk - strongly recommended; copies leaving the host
#     MUST be encrypted.
# The blobs are already AES-GCM encrypted by docassist, but only as long as the key ring
# (deployment/secrets/encryption_keys) is kept - back that up separately, in a secret manager,
# never next to the data.
#
# Environment: COMPOSE (default "docker compose"), BACKUP_AGE_RECIPIENT (optional).
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${root_dir}/backups"
with_storage=1
db_name="docassist"
read -r -a compose <<<"${COMPOSE:-docker compose}"

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || { echo "backup: --output-dir needs a value" >&2; exit 2; }
            output_root=$2
            shift 2
            ;;
        --no-storage) with_storage=0; shift ;;
        -h | --help) usage; exit 0 ;;
        *) echo "backup: unknown argument: $1" >&2; exit 2 ;;
    esac
done

recipient="${BACKUP_AGE_RECIPIENT:-}"
if [[ -n "$recipient" ]]; then
    command -v age >/dev/null 2>&1 || { echo "backup: BACKUP_AGE_RECIPIENT set but age is not installed" >&2; exit 127; }
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
umask 077
target="${output_root}/${stamp}"
mkdir -p -- "$target"
chmod 0700 -- "$output_root" "$target"

cd -- "$root_dir"

# encrypt_to FILE: stdin -> FILE (age-encrypted when a recipient is configured).
encrypt_to() {
    if [[ -n "$recipient" ]]; then
        age --encrypt --recipient "$recipient" --output "$1.age"
    else
        cat >"$1"
    fi
}

echo "backup: database ${db_name} -> ${target}" >&2
# The superuser password is read inside the container from its secret file; it never
# appears on this host's command line or environment.
"${compose[@]}" exec -T postgres sh -ceu '
    PGPASSWORD="$(cat /run/secrets/postgres_superuser_password)"
    export PGPASSWORD
    exec pg_dump --username=postgres --dbname="$1" --format=custom --compress=6 --no-password
' pg_dump "$db_name" | encrypt_to "${target}/database.dump"

if [[ $with_storage -eq 1 ]]; then
    echo "backup: object store -> ${target}" >&2
    # A throw-away container of the api service (same image, same volume, read-only).
    "${compose[@]}" run --rm -T --no-deps --entrypoint sh api -ceu \
        'cd /var/lib/docassist && mkdir -p storage && exec tar -czf - storage' \
        | encrypt_to "${target}/storage.tar.gz"
fi

cat >"${target}/MANIFEST" <<EOF
created_utc=${stamp}
database=${db_name}
storage_included=${with_storage}
encrypted=$([[ -n "$recipient" ]] && echo age || echo no)
EOF

(
    cd -- "$target"
    files=(MANIFEST database.dump*)
    if [[ $with_storage -eq 1 ]]; then
        files+=(storage.tar.gz*)
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- "${files[@]}" >SHA256SUMS
    else
        shasum -a 256 -- "${files[@]}" >SHA256SUMS
    fi
)

echo "backup: done: ${target}" >&2
if [[ -z "$recipient" ]]; then
    echo "backup: WARNING: files are NOT encrypted (set BACKUP_AGE_RECIPIENT)" >&2
fi
echo "backup: verify with: scripts/restore.sh --drill ${target}" >&2
