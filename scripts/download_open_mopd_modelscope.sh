#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'USAGE'
Download the Open-MOPD dataset and training checkpoints from ModelScope.

Usage:
  bash scripts/download_open_mopd_modelscope.sh [ASSET_ROOT]

Defaults:
  ASSET_ROOT=<repo>/ms_sync
  DATA_DIR=<ASSET_ROOT>/data
  MODEL_ROOT=<ASSET_ROOT>/models

To reproduce a ./data and ./models layout from the current directory:
  bash scripts/download_open_mopd_modelscope.sh .

Environment overrides:
  ASSET_ROOT       output root when no positional argument is given
  DATA_DIR         dataset destination
  MODEL_ROOT       model destination root
  MODELSCOPE_BIN   ModelScope CLI executable (default: modelscope)
  DOWNLOAD_FINAL   set to 1 to also download the released final student
USAGE
}

if (($# > 1)); then
    usage >&2
    exit 2
fi

if (($# == 1)) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

ASSET_ROOT="${1:-${ASSET_ROOT:-${REPO_ROOT}/ms_sync}}"
DATA_DIR="${DATA_DIR:-${ASSET_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${ASSET_ROOT}/models}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-modelscope}"
DOWNLOAD_FINAL="${DOWNLOAD_FINAL:-0}"

if ! command -v "${MODELSCOPE_BIN}" >/dev/null 2>&1; then
    cat >&2 <<EOF
[modelscope] error: '${MODELSCOPE_BIN}' was not found.
Install ModelScope in the environment where you will download the assets:
  python3 -m pip install modelscope
EOF
    exit 127
fi

case "${DOWNLOAD_FINAL}" in
    0|1) ;;
    *)
        echo "[modelscope] error: DOWNLOAD_FINAL must be 0 or 1" >&2
        exit 2
        ;;
esac

mkdir -p "${DATA_DIR}" "${MODEL_ROOT}"

download_dataset() {
    local repo_id="$1"
    local destination="$2"

    echo "[modelscope] dataset ${repo_id} -> ${destination}"
    if ! "${MODELSCOPE_BIN}" download \
        --dataset "${repo_id}" \
        --local_dir "${destination}"; then
        echo "[modelscope] error: failed to download dataset ${repo_id}" >&2
        return 1
    fi
}

download_model() {
    local repo_id="$1"
    local destination="$2"

    echo "[modelscope] model ${repo_id} -> ${destination}"
    if ! "${MODELSCOPE_BIN}" download \
        --model "${repo_id}" \
        --local_dir "${destination}"; then
        echo "[modelscope] error: failed to download model ${repo_id}" >&2
        return 1
    fi
}

require_file() {
    local path="$1"
    [[ -s "${path}" ]] || {
        echo "[modelscope] error: expected downloaded file is missing or empty: ${path}" >&2
        return 1
    }
}

verify_model() {
    local path="$1"
    local weights=()

    require_file "${path}/config.json"
    shopt -s nullglob
    weights=("${path}"/*.safetensors)
    shopt -u nullglob
    if ((${#weights[@]} == 0)); then
        echo "[modelscope] error: no safetensors weights found in ${path}" >&2
        return 1
    fi
}

download_dataset \
    "BytedTsinghua-SIA/Open-MOPD-Data" \
    "${DATA_DIR}"

download_model \
    "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT" \
    "${MODEL_ROOT}/mixsft"
download_model \
    "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math" \
    "${MODEL_ROOT}/rl-math"
download_model \
    "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Code" \
    "${MODEL_ROOT}/rl-code"
download_model \
    "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-IF" \
    "${MODEL_ROOT}/rl-if"

if [[ "${DOWNLOAD_FINAL}" == 1 ]]; then
    download_model \
        "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-Final" \
        "${MODEL_ROOT}/final"
fi

require_file "${DATA_DIR}/rl_prompt_mix/train.parquet"
require_file "${DATA_DIR}/rl_prompt_mix/manifest.json"
verify_model "${MODEL_ROOT}/mixsft"
verify_model "${MODEL_ROOT}/rl-math"
verify_model "${MODEL_ROOT}/rl-code"
verify_model "${MODEL_ROOT}/rl-if"
if [[ "${DOWNLOAD_FINAL}" == 1 ]]; then
    verify_model "${MODEL_ROOT}/final"
fi

echo "[modelscope] download and verification completed"
echo "[modelscope] data:   ${DATA_DIR}"
echo "[modelscope] models: ${MODEL_ROOT}"
