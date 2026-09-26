# UP-MOPD: Update Projection for Multi-Teacher On-Policy Distillation

UP-MOPD combines domain teachers in a shared student by constraining the
parameter update produced by AdamW. The original mixed gradient advances the
optimizer state; a candidate update that violates an active domain's first-order
constraint is projected before it is applied to the model.

This repository contains the **verl implementation for mathematics, code, and
instruction following**, built on the Open-MOPD training and evaluation pipeline.

## Method

For domain gradients collected in `G` and the optimizer's candidate displacement
`delta_0`, UP-MOPD solves

```text
minimize    1/2 ||delta - delta_0||²
subject to Gᵀ delta <= 0
```

The implementation forms a small dual problem from domain-level Gram statistics,
then materializes and verifies the projected update in parameter storage
precision. The optimizer state retains the update computed from the original
mixed gradient.

- [Projection solver and parameter update](training/verl/verl/workers/actor/mopd_projection.py)
- [Actor integration and domain replay](training/verl/verl/workers/actor/dp_actor.py)
- [Teacher routing and domain utilities](training/verl/verl/workers/actor/mt_opd.py)
- [Projection tests](training/verl/tests/test_mopd_gradient_projection.py)

## Installation

Use a Linux environment with NVIDIA GPUs, a compatible CUDA/PyTorch stack,
and the vLLM dependencies described in the [vendored verl README](training/verl/README.md).
Install this checkout's patched verl:

```bash
git clone https://github.com/Tocci-Zhu/UP-MOPD.git
cd UP-MOPD
cd training
bash install_requirements.sh
cd ..
```

## Models and data

The public three-domain setup uses the released Open-MOPD MixSFT student,
RL-Math, RL-Code, and RL-IF teachers, and mixed training prompts. Obtain them
from the [Open-MOPD collection](https://huggingface.co/collections/BytedTsinghua-SIA/open-mopd-multi-teacher-on-policy-distillation)
or use the provided ModelScope downloader:

```bash
python3 -m pip install modelscope
bash scripts/download_open_mopd_modelscope.sh .
```

This creates `models/mixsft`, `models/rl-math`, `models/rl-code`, `models/rl-if`,
and `data/rl_prompt_mix/train.parquet`. Model weights and datasets are kept
outside Git.

## Run UP-MOPD

The main entry point is `scripts/local/up-mopd.sh`. It enables hard update
projection, disables global gradient clipping, sets both random seeds to 1,
and writes checkpoints and JSONL metrics into a dedicated run directory.

```bash
TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=128 \
PPO_MICRO_BATCH_PER_GPU=1 LOG_PROB_MICRO_BATCH_PER_GPU=1 \
RM_MICRO_BATCH_PER_GPU=1 MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=16384 RUN_ID=seed1 \
bash scripts/local/up-mopd.sh \
  --model "$PWD/models/mixsft" \
  --teacher "$PWD/models/rl-math" \
  --teacher "$PWD/models/rl-code" \
  --teacher "$PWD/models/rl-if" \
  --domains math,code,if \
  --train "$PWD/data/rl_prompt_mix/train.parquet" \
  --val "$PWD/data/eval/math/aime24.parquet" \
  --output "$PWD/runs/up-mopd/seed1" \
  --gpus 8
```

Launchers print the command by default. Add `--run` to execute it. The example
sets a shared response-length limit; the full command and supported overrides
are documented in [Local launchers](scripts/local/README.md).

Without `--output`, UP-MOPD writes to `../up-mopd-runs/<RUN_ID>/`. Set
`UP_MOPD_RUN_ROOT` to change that root. Run names use `up-mopd-<RUN_ID>` and the
tracking project is `UP-MOPD`.

The generic M-OPD baseline remains available through `scripts/local/mt_opd.sh`.
The current update-projection integration uses FSDP1 and PyTorch AdamW; supported
training settings are listed in the launcher documentation.

## Evaluation

Use the [evaluation guide](evals/README.md) for offline rollouts and scoring.
The [verifier guide](evals/verifier/README.md) describes benchmark dependencies;
third-party benchmark revisions are recorded in
[`repos.lock.json`](evals/verifier/third_party/repos.lock.json).

```bash
bash scripts/local/eval.sh --help
```

## Repository layout

```text
scripts/local/up-mopd.sh    UP-MOPD training entry point
scripts/local/             Baseline, SFT, RL, OPD, and evaluation launchers
training/verl/             Patched verl framework and projection implementation
training/scripts/          Dataset preparation utilities
evals/                     Rollout generation, verifiers, and score aggregation
experiments/               Data conversion, model merging, analysis, and tests
docs/open-mopd.md           Upstream Open-MOPD README and reference
```

## Validation

Launcher tests run without GPUs or training dependencies:

```bash
python3 -m pip install pytest
python3 -m pytest -q experiments/tests/test_local_shell_entrypoints.py
```

With the training dependencies installed, run the projection tests:

```bash
PYTHONPATH=training/verl python3 -m pytest -q \
  training/verl/tests/test_mopd_gradient_projection.py
```

## Acknowledgments

The training and evaluation pipeline builds on **Open-MOPD** and **verl**.
Public checkpoints and datasets are provided by the Open-MOPD authors.
Their original README and citation are retained in
[docs/open-mopd.md](docs/open-mopd.md). Existing third-party copyright notices
and licenses, including [verl's Apache 2.0 license](training/verl/LICENSE),
remain with their respective components.
