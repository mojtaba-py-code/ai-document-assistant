#!/usr/bin/env bash
# First-start initialisation for the docassist database (runs from
# /docker-entrypoint-initdb.d only when the data directory is empty).
#
# Creates three LOGIN roles, each with its own password read from a Docker secret file:
#
#   docassist_owner   owns the database and every table; used ONLY by migrations
#                     (the "migrate" service). Not a superuser, cannot create roles/databases.
#   docassist_app     the API. DML only (granted by the migration), subject to row-level
#                     security, NOBYPASSRLS, never owns a table.
#   docassist_worker  the background worker. Same restrictions, different grants.
#
# plus the database itself (owned by docassist_owner, CONNECT revoked from PUBLIC) and the
# pgvector extension (created here, as the superuser, because the owner role is not one).
#
# The entrypoint *sources* this file when it lost its exec bit (e.g. a Windows checkout), so
# the body runs in a subshell: its shell options and variables cannot leak into the entrypoint.
(
    set -euo pipefail

    db_name="${DOCASSIST_DB_NAME:-docassist}"
    secrets_dir="${DOCASSIST_DB_SECRETS_DIR:-/run/secrets}"

    if [[ ! "$db_name" =~ ^[a-z][a-z0-9_]{0,62}$ ]]; then
        echo "docassist-init: DOCASSIST_DB_NAME must be a lowercase SQL identifier" >&2
        exit 1
    fi
    for name in postgres_owner_password postgres_app_password postgres_worker_password; do
        if [[ ! -s "${secrets_dir}/${name}" ]]; then
            echo "docassist-init: missing secret file ${secrets_dir}/${name}" >&2
            exit 1
        fi
    done

    # Passwords never appear on a command line or in the environment: psql reads them from
    # the files itself (\set with backquotes) and binds them as quoted literals (:'var').
    # log_min_error_statement=panic keeps a failed statement - which could carry a password -
    # out of the server log.
    psql --no-psqlrc -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
        -v db_name="$db_name" -v secrets_dir="$secrets_dir" <<'SQL'
SET log_min_error_statement = panic;
SET log_statement = 'none';
SET password_encryption = 'scram-sha-256';

\set owner_file :secrets_dir '/postgres_owner_password'
\set app_file :secrets_dir '/postgres_app_password'
\set worker_file :secrets_dir '/postgres_worker_password'
\set owner_password `cat :'owner_file'`
\set app_password `cat :'app_file'`
\set worker_password `cat :'worker_file'`

CREATE ROLE docassist_owner LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    CONNECTION LIMIT 10 PASSWORD :'owner_password';
CREATE ROLE docassist_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    NOINHERIT CONNECTION LIMIT 200 PASSWORD :'app_password';
CREATE ROLE docassist_worker LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    NOINHERIT CONNECTION LIMIT 100 PASSWORD :'worker_password';
\unset owner_password
\unset app_password
\unset worker_password

CREATE DATABASE :"db_name" OWNER docassist_owner TEMPLATE template0 ENCODING 'UTF8';
REVOKE ALL ON DATABASE :"db_name" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"db_name" TO docassist_app, docassist_worker;

\connect :db_name
CREATE EXTENSION IF NOT EXISTS vector;
-- PostgreSQL 15+ already denies CREATE on "public" to PUBLIC; state it explicitly anyway.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
SQL
    echo "docassist-init: database ${db_name} and roles docassist_owner/app/worker created"
)
