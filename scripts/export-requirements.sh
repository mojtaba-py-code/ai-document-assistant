#!/bin/sh
# Export the hash-pinned *runtime* requirements of docassist (no development tools, not the
# project itself). One code path feeds the container build, pip-audit in CI and `make security`.
#
#   scripts/export-requirements.sh [-o FILE] [--extra NAME]... [--all-extras]
#
# * uv.lock committed  -> `uv export --locked`: byte-for-byte reproducible, fails if the lock
#                         file is stale.
# * no uv.lock         -> `uv pip compile pyproject.toml`: resolves the declared ranges now.
#
# Every line carries --hash pins, so the consumer installs with --require-hashes and a
# tampered or substituted distribution is rejected.
set -eu

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; }

output="requirements.txt"
extra_args=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        -o | --output)
            [ "$#" -ge 2 ] || { echo "export-requirements: $1 needs a value" >&2; exit 2; }
            output=$2
            shift 2
            ;;
        --extra)
            [ "$#" -ge 2 ] || { echo "export-requirements: --extra needs a value" >&2; exit 2; }
            case "$2" in
                "" | *[!a-z0-9-]*)
                    echo "export-requirements: invalid extra name" >&2
                    exit 2
                    ;;
            esac
            extra_args="$extra_args --extra $2"
            shift 2
            ;;
        --all-extras)
            extra_args="$extra_args --all-extras"
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "export-requirements: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

command -v uv >/dev/null 2>&1 || {
    echo "export-requirements: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 127
}

# extra_args only ever holds validated extra names and fixed flags: word splitting is intended.
if [ -f uv.lock ]; then
    # shellcheck disable=SC2086
    uv export --locked --no-dev --no-emit-project --format requirements-txt \
        --quiet $extra_args --output-file "$output"
else
    # shellcheck disable=SC2086
    uv pip compile pyproject.toml --generate-hashes --no-header --quiet \
        $extra_args --output-file "$output"
fi

echo "export-requirements: wrote $output" >&2
