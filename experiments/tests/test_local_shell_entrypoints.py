"""Smoke tests for the supported local shell entry points.

These tests intentionally use ``/bin/echo`` instead of starting a trainer.  They
verify the user-facing contract (shell syntax, local-path validation, and command
construction) without requiring GPUs or model/data downloads.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = ("opd", "rl", "sft", "eval", "mt_opd")


def _run(
    name: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ROOT / "scripts/local" / f"{name}.sh"), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_launchers_are_valid_shell() -> None:
    for name in (*LAUNCHERS, "up-mopd"):
        result = subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/local" / f"{name}.sh")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_local_launchers_print_commands_without_running(tmp_path: Path) -> None:
    model = tmp_path / "model"
    teacher = tmp_path / "teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    model.mkdir()
    teacher.mkdir()
    train.touch()
    val.touch()

    base = (
        "--model",
        str(model),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--torchrun",
        "/bin/echo",
        "--dry-run",
    )
    for name in LAUNCHERS[:-1]:
        result = _run(name, *base)
        assert result.returncode == 0, (name, result.stderr)
        assert "[local]" in result.stdout
        assert "://" not in result.stdout
        assert "remote_submit" not in result.stdout

    result = _run(
        "mt_opd",
        *base,
        "--teacher",
        str(teacher),
        "--teacher",
        str(teacher),
        "--domains",
        "math,code",
    )
    assert result.returncode == 0, result.stderr
    assert "://" not in result.stdout
    assert "remote_submit" not in result.stdout


def test_default_and_up_mopd_build_expected_three_domain_commands(tmp_path: Path) -> None:
    model = tmp_path / "model"
    teachers = [tmp_path / f"teacher-{index}" for index in range(3)]
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    model.mkdir()
    for teacher in teachers:
        teacher.mkdir()
    train.touch()
    val.touch()

    args = [
        "--model",
        str(model),
        "--train",
        str(train),
        "--val",
        str(val),
        "--python",
        "/bin/echo",
        "--dry-run",
    ]
    for teacher in teachers:
        args.extend(("--teacher", str(teacher)))
    args.extend(("--domains", "math,code,if"))

    commands = {}
    for launcher, method in (
        ("mt_opd", "default"),
        ("up-mopd", "up-mopd"),
    ):
        output = tmp_path / method
        env = os.environ.copy()
        env.update(
            {
                "RUN_ID": "test",
                "OUTPUT_DIR": str(output),
                "DATA_SEED": "1",
                "ROLLOUT_SEED": "1",
                "RESUME_MODE": "disable",
            }
        )
        env.pop("MOPD_GRADIENT_PROJECTION_DECOMPOSITION_RTOL", None)
        result = _run(launcher, *args, env=env)
        assert result.returncode == 0, result.stderr
        commands[launcher] = result.stdout
        assert "actor_rollout_ref.actor.ppo_mini_batch_size=32" in result.stdout
        assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4" in result.stdout
        assert "+actor_rollout_ref.rollout.log_prob_top_k=256" in result.stdout
        assert "data.seed=1" in result.stdout
        assert "+actor_rollout_ref.rollout.seed=1" in result.stdout
        assert "trainer.resume_mode=disable" in result.stdout
        assert str(output) in result.stdout
        assert not output.exists(), "dry-run must not create experiment outputs"

    assert "actor_rollout_ref.actor.grad_clip=1.0" in commands["mt_opd"]
    assert "mopd_gradient_projection_mode=none" in commands["mt_opd"]
    assert (
        "mopd_gradient_projection_decomposition_rtol=1e-3" in commands["mt_opd"]
    )
    assert (
        "actor_rollout_ref.actor.grad_clip=0.0" in commands["up-mopd"]
    )
    assert (
        "mopd_gradient_projection_mode=adam_project_hard"
        in commands["up-mopd"]
    )
    assert (
        "mopd_gradient_projection_decomposition_rtol=1e-2"
        in commands["up-mopd"]
    )

    override_env = os.environ.copy()
    override_env.update(
        {
            "RUN_ID": "test-override",
            "OUTPUT_DIR": str(tmp_path / "up-mopd-override"),
            "MOPD_GRADIENT_PROJECTION_DECOMPOSITION_RTOL": "5e-3",
        }
    )
    override_result = _run(
        "up-mopd",
        *args,
        env=override_env,
    )
    assert override_result.returncode == 0, override_result.stderr
    assert (
        "mopd_gradient_projection_decomposition_rtol=5e-3"
        in override_result.stdout
    )


def test_run_mode_rejects_remote_uri(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    result = _run(
        "rl",
        "--model",
        "https://example.invalid/model",
        "--train",
        str(tmp_path / "train.parquet"),
        "--val",
        str(tmp_path / "val.parquet"),
        "--run",
    )
    assert result.returncode != 0
    assert "local filesystem path" in result.stderr


def test_up_mopd_run_names_and_output_root(tmp_path: Path) -> None:
    env = os.environ.copy()
    for key in ("OUTPUT_DIR", "CHECKPOINT_DIR", "PROJECT_NAME", "EXPERIMENT_NAME"):
        env.pop(key, None)
    env.update({"RUN_ID": "seed1", "UP_MOPD_RUN_ROOT": str(tmp_path / "runs")})
    result = _run("up-mopd", "--dry-run", env=env)
    assert result.returncode == 0, result.stderr
    assert "trainer.project_name=UP-MOPD" in result.stdout
    assert "trainer.experiment_name=up-mopd-seed1" in result.stdout
    assert f"trainer.default_local_dir={tmp_path}/runs/seed1/checkpoints" in result.stdout
    assert not (tmp_path / "runs").exists()


def test_up_mopd_output_override_routes_checkpoints_and_metrics(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    train = tmp_path / "train.parquet"
    train.touch()
    output = tmp_path / "selected-output"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$VERL_FILE_LOGGER_PATH" "$@"\n')
    fake_python.chmod(0o755)
    env = os.environ.copy()
    for key in ("OUTPUT_DIR", "CHECKPOINT_DIR", "VERL_FILE_LOGGER_PATH"):
        env.pop(key, None)
    env.update({"RUN_ID": "seed1", "UP_MOPD_RUN_ROOT": str(tmp_path / "unused-root")})
    args = (
        "--model", str(model), "--teacher", str(model), "--teacher", str(model),
        "--domains", "math,code", "--train", str(train), "--val", str(train),
        "--output", str(output), "--python", str(fake_python), "--run",
    )
    result = _run("up-mopd", *args, env=env)
    assert result.returncode == 0, result.stderr
    assert str(output / "metrics.jsonl") in result.stdout
    assert f"trainer.default_local_dir={output}/checkpoints" in result.stdout
    assert (output / "checkpoints").is_dir()
    assert not (tmp_path / "unused-root").exists()

    env["CHECKPOINT_DIR"] = str(tmp_path / "custom-checkpoints")
    env["VERL_FILE_LOGGER_PATH"] = str(tmp_path / "custom-metrics.jsonl")
    result = _run("up-mopd", *args, env=env)
    assert result.returncode == 0, result.stderr
    assert f"trainer.default_local_dir={env['CHECKPOINT_DIR']}" in result.stdout
    assert env["VERL_FILE_LOGGER_PATH"] in result.stdout


def test_run_mode_executes_with_local_paths_and_fake_binaries(tmp_path: Path) -> None:
    model = tmp_path / "model"
    teacher = tmp_path / "teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    model.mkdir()
    teacher.mkdir()
    train.touch()
    val.touch()

    base = (
        "--model",
        str(model),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--torchrun",
        "/bin/echo",
        "--run",
    )
    for name in LAUNCHERS[:-1]:
        result = _run(name, *base)
        assert result.returncode == 0, (name, result.stderr)
        assert "[local]" in result.stdout

    result = _run(
        "mt_opd",
        *base,
        "--teacher",
        str(teacher),
        "--teacher",
        str(teacher),
        "--domains",
        "math,code",
    )
    assert result.returncode == 0, result.stderr
