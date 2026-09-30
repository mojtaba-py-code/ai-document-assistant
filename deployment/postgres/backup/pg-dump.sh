#!/usr/bin/env bash
# Logical backup of the docassist database from any PostgreSQL (managed service, VM or the
# compose container), for hosts that have the PostgreSQL 16 client tools.
#
#   PGSERVICEFILE=/etc/docassist/pg_service.conf deployment/postgres/backup/pg-dump.sh [OUTPUT_DIR]
#
# Connection details come from a libpq service file (service "docassist-backup") and the
# password from a ~/.pgpass-style file (PGPASSFILE, mode 0600) - never from the command line,
# so they do not show up in `ps` or shell history. Example service file:
#
#   [docassist-backup]
#   host=db.internal.example.com
#   port=5432
#   dbname=docassist
#   user=docassist_owner        # owns every table: dumps without row-level-security gaps
#   sslmode=verify-full
#   sslrootcert=/etc/docassist/db-ca.crt
#
# Output: <OUTPUT_DIR>/docassist-<UTC timestamp>.dump (custom format, restorable with
# pg_restore, table by table if needed) plus a .sha256 file. Encrypt before shipping it
# off-host (e.g. `age -r <recipient>`): the dump contains document text.
#
# This is a point-in-time copy; for point-in-time *recovery* (RPO of minutes) use
# pg-basebackup.sh together with WAL archiving (postgresql.wal-archiving.conf).
set -euo pipefail

out_dir="${1:-./backups}"
service="${DOCASSIST_BACKUP_SERVICE:-docassist-backup}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

command -v pg_dump >/dev/null 2>&1 || { echo "pg-dump: pg_dump (PostgreSQL client) is required" >&2; exit 127; }

umask 077
mkdir -p -- "$out_dir"
file="${out_dir}/docassist-${stamp}.dump"

# --no-password: fail instead of prompting when the password file is missing.
pg_dump --dbname="service=${service}" --format=custom --compress=6 --no-password \
    --lock-wait-timeout=60s --file="$file"

if command -v sha256sum >/dev/null 2>&1; then
    (cd -- "$out_dir" && sha256sum -- "$(basename -- "$file")" >"$(basename -- "$file").sha256")
else
    (cd -- "$out_dir" && shasum -a 256 -- "$(basename -- "$file")" >"$(basename -- "$file").sha256")
fi

# A dump is only a backup once it has been read back: list its table of contents.
pg_restore --list "$file" >/dev/null
echo "pg-dump: wrote ${file}" >&2
