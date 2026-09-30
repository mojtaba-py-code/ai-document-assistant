#!/usr/bin/env bash
# Generate the Docker secret files used by compose.yaml (and the deployment/compose/*.yaml
# overrides) into deployment/secrets/.
#
#   scripts/generate-secrets.sh [--dir DIR] [--with-tls] [--strict]
#
# * Idempotent: an existing file is NEVER overwritten (rotating encryption_keys or the audit
#   key would make stored documents or the audit chain unverifiable). Delete a file on
#   purpose to regenerate it; derived files (DSNs, the Redis ACL, the Qdrant config) are
#   rebuilt from the password files that already exist.
# * Nothing secret is printed: the output lists file names only.
# * The directory is created 0700 (only you can list or open anything inside it). Files are
#   0644 by default because Docker Compose bind-mounts file secrets with their host owner and
#   mode, and the containers run as fixed unprivileged uids (10001 app, 999 postgres/redis,
#   1000 qdrant) that must be able to read them; the 0700 directory keeps other host users out.
#   --strict makes the files 0600 instead (Docker Desktop, rootless Docker, or when you chown
#   the files to the consuming uid yourself).
# * --with-tls also creates a self-signed CA and a server certificate for PostgreSQL
#   (needed by deployment/compose/production.yaml, which requires TLS to the database).
#
# Requires: bash, openssl.
set -euo pipefail

readonly DB_NAME="docassist"
readonly DB_HOST="postgres"
readonly REDIS_HOST="redis"

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
secrets_dir="${root_dir}/deployment/secrets"
with_tls=0
file_mode=0644

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            [[ $# -ge 2 ]] || { echo "generate-secrets: --dir needs a value" >&2; exit 2; }
            secrets_dir=$2
            shift 2
            ;;
        --with-tls) with_tls=1; shift ;;
        --strict) file_mode=0600; shift ;;
        -h | --help) usage; exit 0 ;;
        *) echo "generate-secrets: unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v openssl >/dev/null 2>&1 || { echo "generate-secrets: openssl is required" >&2; exit 127; }

umask 077
mkdir -p -- "$secrets_dir"
chmod 0700 -- "$secrets_dir"

created=()
kept=()

# --- random material (no quotes, no URL-reserved characters) ------------------------------
# (tr -d '\r' as well: a native Windows openssl under Git Bash ends its output with CRLF)
rand_hex() { openssl rand -hex "$1" | tr -d '\r\n'; }
rand_b64() { openssl rand -base64 "$1" | tr -d '\r\n'; }
rand_urlsafe() { rand_b64 "$1" | tr -d '=' | tr '+/' '-_'; }
sha256_hex() { printf '%s' "$1" | openssl dgst -sha256 | tr -d '\r\n' | sed 's/^.*= *//'; }

path_of() { printf '%s/%s' "$secrets_dir" "$1"; }

# write_secret NAME VALUE: atomically create NAME with VALUE. Single-line values are written
# without a trailing newline (they end up verbatim in DSNs and keys); multi-line documents
# (ACL, YAML, PEM) keep a final newline.
write_secret() {
    local name=$1 value=$2 target tmp
    target="$(path_of "$name")"
    tmp="$(mktemp "${secrets_dir}/.${name}.XXXXXX")"
    if [[ $value == *$'\n'* ]]; then
        printf '%s\n' "$value" >"$tmp"
    else
        printf '%s' "$value" >"$tmp"
    fi
    chmod "$file_mode" "$tmp"
    mv -f -- "$tmp" "$target"
    created+=("$name")
}

# ensure NAME GENERATOR...: create NAME from the generator's output unless it already exists.
ensure() {
    local name=$1
    shift
    if [[ -s "$(path_of "$name")" ]]; then
        kept+=("$name")
        return 0
    fi
    write_secret "$name" "$("$@")"
}

read_secret() { tr -d '\r\n' <"$(path_of "$1")"; }

# --- PostgreSQL: one password per role, DSNs derived from them ----------------------------
ensure postgres_superuser_password rand_hex 32
ensure postgres_owner_password rand_hex 32
ensure postgres_app_password rand_hex 32
ensure postgres_worker_password rand_hex 32

dsn() { # dsn ROLE PASSWORD_FILE
    printf 'postgresql+asyncpg://%s:%s@%s:5432/%s' "$1" "$(read_secret "$2")" "$DB_HOST" "$DB_NAME"
}
ensure database_url dsn docassist_app postgres_app_password
ensure database_worker_url dsn docassist_worker postgres_worker_password
ensure database_migration_url dsn docassist_owner postgres_owner_password

# --- Redis: the ACL file holds only the SHA-256 of the password --------------------------
ensure redis_password rand_hex 32
redis_url() { printf 'redis://docassist:%s@%s:6379/0' "$(read_secret redis_password)" "$REDIS_HOST"; }
redis_acl() {
    local digest
    digest="$(sha256_hex "$(read_secret redis_password)")"
    # "default" (unauthenticated) may only PING - used by the container health check.
    printf 'user default on nopass resetkeys resetchannels -@all +ping\n'
    # The application: its own key prefix only, no pub/sub, no dangerous/admin commands.
    printf 'user docassist on #%s resetkeys ~docassist:* resetchannels +@all -@dangerous\n' "$digest"
}
ensure redis_url redis_url
ensure redis_users.acl redis_acl

# --- application keys ---------------------------------------------------------------------
ensure jwt_signing_key rand_urlsafe 48
ensure token_pepper rand_urlsafe 48
ensure audit_hmac_key rand_urlsafe 48
ensure metrics_token rand_urlsafe 32

key_id() { printf 'k%s' "$(date -u +%Y%m%d)"; }
encryption_keys() { printf '%s:%s' "$(key_id)" "$(rand_b64 32)"; }
active_key_id() { # the newest (last) key id in encryption_keys
    read_secret encryption_keys | tr ',' '\n' | sed -n 's/^ *\([A-Za-z0-9_-]*\):.*$/\1/p' | tail -n 1
}
ensure encryption_keys encryption_keys
ensure encryption_key_id active_key_id

# --- Qdrant (optional profile): the server reads the key from its config file -------------
ensure qdrant_api_key rand_urlsafe 32
qdrant_config() {
    printf 'service:\n  api_key: "%s"\ntelemetry_disabled: true\n' "$(read_secret qdrant_api_key)"
}
ensure qdrant_config.yaml qdrant_config

# --- PostgreSQL TLS (production-like override) --------------------------------------------
if [[ $with_tls -eq 1 ]]; then
    if [[ -s "$(path_of postgres_tls.crt)" && -s "$(path_of postgres_tls.key)" ]]; then
        kept+=(postgres_tls_ca.crt postgres_tls.crt postgres_tls.key)
    else
        work="$(mktemp -d)"
        trap 'rm -rf -- "$work"' EXIT
        # Subjects come from config files, not -subj "/CN=...": Git Bash would rewrite that
        # argument into a Windows path.
        cat >"$work/ca.cnf" <<'EOF'
[req]
prompt = no
distinguished_name = dn
x509_extensions = v3_ca
[dn]
CN = docassist-postgres-ca
[v3_ca]
basicConstraints = critical, CA:true, pathlen:0
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
EOF
        cat >"$work/server.cnf" <<EOF
[req]
prompt = no
distinguished_name = dn
[dn]
CN = ${DB_HOST}
[v3_server]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = DNS:${DB_HOST}, DNS:localhost
EOF
        {
            openssl req -x509 -newkey rsa:3072 -nodes -days 825 -sha256 \
                -config "$work/ca.cnf" -keyout "$work/ca.key" -out "$work/ca.crt"
            openssl req -new -newkey rsa:3072 -nodes -sha256 -config "$work/server.cnf" \
                -keyout "$work/server.key" -out "$work/server.csr"
            openssl x509 -req -in "$work/server.csr" -CA "$work/ca.crt" -CAkey "$work/ca.key" \
                -CAcreateserial -days 825 -sha256 -extfile "$work/server.cnf" \
                -extensions v3_server -out "$work/server.crt"
        } >"$work/openssl.log" 2>&1 || {
            echo "generate-secrets: openssl failed:" >&2
            cat "$work/openssl.log" >&2
            exit 1
        }
        write_secret postgres_tls_ca.crt "$(cat "$work/ca.crt")"
        write_secret postgres_tls.crt "$(cat "$work/server.crt")"
        write_secret postgres_tls.key "$(cat "$work/server.key")"
        # The CA private key is not kept: nothing else should ever be signed with it.
    fi
fi

echo "generate-secrets: directory ${secrets_dir} (mode 0700, files ${file_mode})"
if [[ ${#created[@]} -gt 0 ]]; then
    echo "  created: ${created[*]}"
fi
if [[ ${#kept[@]} -gt 0 ]]; then
    echo "  kept (already present): ${kept[*]}"
fi
echo "  not generated - provide them yourself when an override needs them:"
echo "    anthropic_api_key, embedding_api_key   (printf '%s' \"\$KEY\" > <dir>/<name>)"
