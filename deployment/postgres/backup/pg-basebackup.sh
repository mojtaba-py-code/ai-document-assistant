#!/usr/bin/env bash
# Physical base backup (whole cluster) for point-in-time recovery of a self-managed
# PostgreSQL. Combine with continuous WAL archiving (postgresql.wal-archiving.conf); managed
# services (RDS, Cloud SQL, Azure) provide the same through their own PITR settings instead.
#
#   PGSERVICEFILE=/etc/docassist/pg_service.conf deployment/postgres/backup/pg-basebackup.sh [OUTPUT_DIR]
#
# Requires a dedicated replication role - never an application role:
#
#   CREATE ROLE docassist_backup LOGIN REPLICATION PASSWORD '...';   -- then in pg_hba.conf:
#   hostssl replication docassist_backup <backup host>/32 scram-sha-256
#
# and a libpq service "docassist-basebackup" (host, port, user=docassist_backup,
# sslmode=verify-full, sslrootcert=...) with the password in PGPASSFILE (mode 0600).
#
# Output: <OUTPUT_DIR>/base-<UTC timestamp>/ - a plain copy of the data directory including
# the WAL needed for consistency and the backup manifest, verified end to end with
# pg_verifybackup (PostgreSQL 16 verifies plain-format backups only) before the script
# reports success. Compress/encrypt it afterwards for shipping off-host.
set -euo pipefail

out_dir="${1:-./backups}"
service="${DOCASSIST_BASEBACKUP_SERVICE:-docassist-basebackup}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

for tool in pg_basebackup pg_verifybackup; do
    command -v "$tool" >/dev/null 2>&1 || { echo "pg-basebackup: $tool is required" >&2; exit 127; }
done

umask 077
target="${out_dir}/base-${stamp}"
mkdir -p -- "$target"

# -X stream: the WAL needed to make this backup consistent is included, so it is restorable
# on its own; archived WAL then rolls it forward to any later point in time.
pg_basebackup --dbname="service=${service}" --pgdata="$target" --format=plain \
    --wal-method=stream --checkpoint=fast --label="docassist-${stamp}" \
    --manifest-checksums=SHA256 --no-password --progress

# Every file against the manifest checksums, and the WAL range the backup needs.
pg_verifybackup "$target"
echo "pg-basebackup: wrote and verified ${target}" >&2
