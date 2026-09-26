# Local training and evaluation launchers

The launchers use local model and data paths. They print a command by default;
add `--run` to execute it.

| Entry point | Purpose |
| --- | --- |
| `up-mopd.sh` | UP-MOPD with hard projection of AdamW parameter updates |
| `mt_opd.sh` | Multi-teacher OPD baseline and shared trainer options |
| `opd.sh` | Single-teacher OPD |
| `sft.sh` | Supervised fine-tuning |
| `rl.sh` | Reinforcement learning |
| `eval.sh` | Evaluation rollout generation |

## UP-MOPD

```bash
RUN_ID=seed1 bash scripts/local/up-mopd.sh \
  --model /data/models/mixsft \
  --teacher /data/models/rl-math \
  --teacher /data/models/rl-code \
  --teacher /data/models/rl-if \
  --domains math,code,if \
  --train /data/Open-MOPD-Data/rl_prompt_mix/train.parquet \
  --val /data/Open-MOPD-Data/eval/math/aime24.parquet \
  --gpus 8
```

The wrapper sets `mopd_gradient_projection_mode=adam_project_hard`,
`grad_clip=0`, and `epsilon=0`. Data and rollout seeds default to 1, resuming
is disabled by default, and both console and JSONL logging are enabled.

Outputs use `../up-mopd-runs/<RUN_ID>/`, with `checkpoints/` and `metrics.jsonl`
inside that directory. Set `UP_MOPD_RUN_ROOT` or pass `--output` to select a
location. Explicit `CHECKPOINT_DIR`, `--checkpoint`, and
`VERL_FILE_LOGGER_PATH` overrides are supported. `OPEN_MOPD_RUN_ROOT` remains
accepted as a fallback for existing environment configurations.

The tracking project is `UP-MOPD`, and experiment names use `up-mopd-<RUN_ID>`.
`PROJECT_NAME` and `EXPERIMENT_NAME` can override these names.

## Shared configuration

Environment variables mirror the options: `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`,
`OUTPUT_DIR`, `CHECKPOINT_DIR`, `GPUS`, `NODES`, `PYTHON_BIN`, and `TORCHRUN_BIN`.
For multiple teachers, repeat `--teacher` or set comma-separated
`TEACHER_MODEL_PATHS`; `TEACHER_DOMAINS` supplies the matching domain labels.
Additional Hydra overrides can be passed after `--`.

The shared M-OPD launcher defaults to 32 prompts per batch, one optimizer
mini-batch per rollout batch, 4 samples per GPU micro-batch, a 1,024-token prompt
limit, and a 2,048-token response limit. Override these with
`TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, `PPO_MICRO_BATCH_PER_GPU`,
`MAX_PROMPT_LENGTH`, and `MAX_RESPONSE_LENGTH`. See the root README for a
larger-batch launch example.

For the baseline, use the same paths and configuration with `mt_opd.sh`.
Its default projection mode is `none` and gradient clipping is `1.0`;
set `ACTOR_GRAD_CLIP=0` for a baseline without clipping.

## Update-projection integration

The current integration supports PyTorch AdamW with FSDP1, one PPO epoch,
one optimizer mini-batch per rollout batch, static micro-batches, token-mean
vanilla policy loss, no actor KL or entropy term, no Ulysses sequence
parallelism, and no M4 advantage refresh.

Each active domain is replayed to obtain its gradient contribution. The mixed
gradient advances AdamW once; UP-MOPD projects the candidate displacement and
verifies the materialized parameter target. Single-domain batches take the
ordinary optimizer step. If finite-precision correction fails verification,
the parameters are restored while the optimizer state is retained.

`MOPD_GRADIENT_PROJECTION_DECOMPOSITION_RTOL` controls the gradient-replay
reconstruction check. Its UP-MOPD default is `1e-2` to accommodate BF16
rounding; the generic M-OPD default is `1e-3`. This tolerance is separate from
the hard projection threshold `epsilon=0`. Keep model dropout and router jitter
at zero for deterministic replay.

## Other entry points

```bash
bash scripts/local/opd.sh --help
bash scripts/local/sft.sh --help
bash scripts/local/rl.sh --help
bash scripts/local/eval.sh --help
```

The launchers support one node and use existing local files. The ModelScope
downloader is a separate command, documented in the root README.
