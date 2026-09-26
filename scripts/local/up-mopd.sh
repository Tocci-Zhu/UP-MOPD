#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${UP_MOPD_RUN_ROOT:-${OPEN_MOPD_RUN_ROOT:-${REPO_ROOT}/../up-mopd-runs}}"

export ACTOR_GRAD_CLIP="0.0"
export MOPD_GRADIENT_PROJECTION_MODE="adam_project_hard"
export MOPD_GRADIENT_PROJECTION_EPSILON="0.0"
export MOPD_GRADIENT_PROJECTION_GRAD_OFFLOAD="${MOPD_GRADIENT_PROJECTION_GRAD_OFFLOAD:-cpu}"
# The actor forward is BF16. Separate per-domain backwards and the ordinary
# mixed backward are mathematically additive, but their GEMM/reduction rounding
# is not bitwise identical. Keep this as a sanity gate while allowing the
# expected BF16-scale residual; this does not relax the hard epsilon=0 check on
# the materialized parameter update.
export MOPD_GRADIENT_PROJECTION_DECOMPOSITION_RTOL="${MOPD_GRADIENT_PROJECTION_DECOMPOSITION_RTOL:-1e-2}"
export DATA_SEED="${DATA_SEED:-1}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-1}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export PROJECT_NAME="${PROJECT_NAME:-UP-MOPD}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-up-mopd-${RUN_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/${RUN_ID}}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console','file']}"

exec bash "${SCRIPT_DIR}/mt_opd.sh" "$@"
