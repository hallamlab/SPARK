#!/usr/bin/env bash
set -euo pipefail

# Resolve preferred executable (Nextflow can override via NXF_REAL_MAMBA)
target="${NXF_REAL_MAMBA:-$(command -v mamba || true)}"
if [[ -z "${target}" ]]; then
    target="$(command -v conda || true)"
fi

if [[ -z "${target}" ]]; then
    echo "[nxf_mamba] Unable to locate 'mamba' or 'conda' on PATH" >&2
    exit 127
fi

auto_yes=false
args=()
for arg in "$@"; do
    case "$arg" in
        --yes|-y|--no-spinner)
            auto_yes=true
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

if [[ ${#args[@]} -eq 0 ]]; then
    args+=("--help")
fi

if [[ "$auto_yes" == true ]]; then
    yes | "$target" "${args[@]}"
    exit $?
else
    exec "$target" "${args[@]}"
fi
