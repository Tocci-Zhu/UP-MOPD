"""CPU tests for optimizer-aware MOPD parameter-displacement projection."""

import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.mopd_projection import (
    ProjectionNumericalError,
    gradient_decomposition_metrics,
    materialize_projected_parameter_target,
    optimizer_parameters,
    project_parameter_step_,
    snapshot_gradients,
    snapshot_parameters,
    solve_guarded_update_correction,
    solve_update_projection,
)
from verl.workers.actor.mt_opd import encode_projection_domains
from verl.workers.config.actor import FSDPActorConfig


class _ToyProjectionActor(DataParallelPPOActor):
    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        top_k=0,
        student_top_k_ids=None,
    ):
        del temperature, calculate_entropy, top_k, student_top_k_ids
        log_probs = self.actor_module(micro_batch["toy_features"]).squeeze(-1)
        return None, log_probs, None, None


def _distributed_projection_worker(
    rank,
    world_size,
    init_method,
    before,
    raw_delta,
    domain_gradients,
    expected_parameter,
) -> None:
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    try:
        shard = slice(rank * 2, (rank + 1) * 2)
        local_before = before[shard].clone()
        parameter = torch.nn.Parameter((local_before + raw_delta[shard]).clone())
        local_domain_gradients = [[gradient[shard].clone()] for gradient in domain_gradients]
        result = project_parameter_step_(
            [parameter],
            [local_before],
            local_domain_gradients,
            chunk_numel=1,
            process_group=dist.group.WORLD,
        )
        torch.testing.assert_close(parameter.detach(), expected_parameter[shard], rtol=0.0, atol=0.0)
        assert all(harm <= 0.0 for harm in result.committed_harms)
    finally:
        dist.destroy_process_group()


def test_projection_domain_encoding_uses_only_globally_present_domains() -> None:
    domain_ids, active_domains = encode_projection_domains(
        ["math", "math"],
        ["math", "code", "if"],
        device="cpu",
    )

    assert active_domains == ["math"]
    torch.testing.assert_close(domain_ids, torch.zeros(2, dtype=torch.long))


def test_projection_domain_encoding_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="without routed teachers"):
        encode_projection_domains(["math", "unknown"], ["math", "code", "if"], device="cpu")


def test_hard_projection_config_rejects_nonzero_epsilon() -> None:
    with pytest.raises(ValueError, match="requires.*epsilon=0"):
        FSDPActorConfig(
            ppo_micro_batch_size_per_gpu=1,
            mopd_gradient_projection_mode="adam_project_hard",
            mopd_gradient_projection_epsilon=0.1,
        )


def test_three_domain_active_set_projection() -> None:
    result = solve_update_projection(
        gradient_gram=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        nominal_harms=[1.0, -2.0, 3.0],
    )

    assert result.correction_coefficients == pytest.approx((1.0, 0.0, 3.0))
    assert result.projected_harms == pytest.approx((0.0, -2.0, 0.0))
    assert result.active_set == (0, 2)


def test_projection_is_invariant_to_positive_domain_gradient_scaling() -> None:
    gradients = torch.tensor([[1.0, 2.0], [-2.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    delta = torch.tensor([0.7, -0.3], dtype=torch.float64)
    gram = gradients @ gradients.T
    harms = gradients @ delta
    original = solve_update_projection(gram.tolist(), harms.tolist())

    scales = torch.tensor([3.0, 0.25, 5.0], dtype=torch.float64)
    scaled_gradients = gradients * scales.unsqueeze(-1)
    scaled = solve_update_projection(
        (scaled_gradients @ scaled_gradients.T).tolist(),
        (scaled_gradients @ delta).tolist(),
    )

    original_correction = sum(
        coefficient * gradient
        for coefficient, gradient in zip(original.correction_coefficients, gradients, strict=True)
    )
    scaled_correction = sum(
        coefficient * gradient
        for coefficient, gradient in zip(scaled.correction_coefficients, scaled_gradients, strict=True)
    )
    torch.testing.assert_close(original_correction, scaled_correction)


def test_bad_projection_gram_fails_loudly() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        solve_update_projection([[1.0, 2.0], [2.0, 1.0]], [1.0, 1.0])


def test_near_opposite_constraints_remain_solvable() -> None:
    cross = -(1.0 - 1e-8)
    result = solve_update_projection(
        [[1.0, cross], [cross, 1.0]],
        [1.0, 1.000001],
    )

    assert max(result.projected_harms) <= 1e-7


def test_unreliable_near_singular_opposite_constraints_fall_back() -> None:
    cross = -(1.0 - 1e-10)
    with pytest.raises(ProjectionNumericalError):
        solve_update_projection(
            [[1.0, cross], [cross, 1.0]],
            [1.0, 1.000001],
        )


def test_near_parallel_constraints_remain_solvable() -> None:
    cross = 1.0 - 1e-8
    result = solve_update_projection(
        [[1.0, cross], [cross, 1.0]],
        [1.0, 1.0 + 0.5e-8],
    )

    assert result.dual_variables == pytest.approx((0.25, 0.75), abs=1e-6)
    assert max(result.projected_harms) <= 1e-8


def test_three_domain_solver_handles_redundant_collinear_constraint() -> None:
    gradients = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    delta = torch.tensor([1.0, 1.0], dtype=torch.float64)
    result = solve_update_projection(
        (gradients @ gradients.T).tolist(),
        (gradients @ delta).tolist(),
    )

    assert max(result.projected_harms) <= 1e-8


def test_zero_gradient_cannot_have_nonzero_harm() -> None:
    with pytest.raises(ValueError, match="zero domain gradient"):
        solve_update_projection([[0.0]], [1.0])
    with pytest.raises(ProjectionNumericalError, match="zero domain gradient"):
        solve_guarded_update_correction([[0.0]], [1e-12], guard_multiplier=2.0)


def test_projected_target_forms_delta_before_rounding_absolute_fp32_parameter() -> None:
    parameter = torch.tensor([0.13913585245609283, 15.601109504699707], dtype=torch.float32)
    gradient = torch.tensor([-0.002634183270856738, -1.715383041300811e-05], dtype=torch.float32)
    nominal_delta = torch.tensor([2.208051341767714e-08, -0.0010636691004037857], dtype=torch.float32)
    correction = 0.002621021723228111

    left_associated_target = parameter + nominal_delta - correction * gradient
    left_associated_delta = left_associated_target.double() - parameter.double()
    left_associated_harm = torch.dot(gradient.double(), left_associated_delta).item()
    materialization = materialize_projected_parameter_target(
        parameter,
        nominal_delta,
        (gradient,),
        (correction,),
    )
    stable_harm = torch.dot(gradient.double(), materialization.materialized_delta).item()

    assert left_associated_harm > 0.0
    assert stable_harm <= 0.0


def test_domain_contributions_reconstruct_the_raw_gradient() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    domain_1 = torch.tensor([1.0, 0.0, -2.0, 3.0])
    domain_2 = torch.tensor([-0.5, 2.0, 1.0, -1.0])
    domain_3 = torch.tensor([0.25, -1.0, 0.0, 0.5])
    parameter.grad = domain_1 + domain_2 + domain_3

    error_norm, raw_norm, relative_error = gradient_decomposition_metrics(
        [parameter],
        [[domain_1], [domain_2], [domain_3]],
        chunk_numel=2,
    )

    assert error_norm == pytest.approx(0.0)
    assert raw_norm > 0.0
    assert relative_error == pytest.approx(0.0)


def test_real_adamw_step_advances_state_but_commits_projected_parameter() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=0.1,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.0,
    )
    domain_1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    domain_2 = torch.tensor([-2.0, 1.0], dtype=torch.float32)
    raw_gradient = domain_1 + domain_2
    parameter.grad = raw_gradient.clone()
    before = snapshot_parameters([parameter], offload="device")

    optimizer.step()
    raw_candidate = parameter.detach().clone()
    result = project_parameter_step_(
        [parameter],
        before,
        [[domain_1], [domain_2]],
        chunk_numel=1,
    )

    committed_delta = parameter.detach().double() - before[0].double()
    assert torch.dot(domain_1.double(), committed_delta).item() <= 0.0
    assert torch.dot(domain_2.double(), committed_delta).item() <= 0.0
    assert not torch.equal(parameter.detach(), raw_candidate)
    assert optimizer.state[parameter]["step"].item() == pytest.approx(1.0)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], raw_gradient)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], raw_gradient.square())
    assert not result.zero_fallback


def test_zero_grad_clip_means_disabled_not_zeroed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    module = torch.nn.Linear(2, 1, bias=False)
    reference = torch.nn.Linear(2, 1, bias=False)
    reference.load_state_dict(module.state_dict())
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1, weight_decay=0.0)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.1, weight_decay=0.0)
    config = FSDPActorConfig(
        ppo_micro_batch_size_per_gpu=1,
        grad_clip=0.0,
        use_torch_compile=False,
    )
    actor = DataParallelPPOActor(config=config, actor_module=module, actor_optimizer=optimizer)
    inputs = torch.tensor([[1.0, -2.0]])
    module(inputs).sum().backward()
    reference(inputs).sum().backward()
    before = module.weight.detach().clone()

    grad_norm = actor._optimizer_step()
    reference_optimizer.step()

    assert torch.is_tensor(grad_norm)
    assert actor._last_mopd_projection_metrics == {}
    assert not torch.equal(module.weight.detach(), before)
    torch.testing.assert_close(module.weight.detach(), reference.weight.detach())


def test_single_active_domain_bypasses_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    module = torch.nn.Linear(2, 1, bias=False)
    reference = torch.nn.Linear(2, 1, bias=False)
    reference.load_state_dict(module.state_dict())
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1, weight_decay=0.0)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.1, weight_decay=0.0)
    config = FSDPActorConfig(
        ppo_micro_batch_size_per_gpu=1,
        grad_clip=0.0,
        use_torch_compile=False,
    )
    actor = DataParallelPPOActor(config=config, actor_module=module, actor_optimizer=optimizer)
    actor._mopd_projection_mode = "adam_project_hard"
    inputs = torch.tensor([[1.0, -2.0]])
    module(inputs).sum().backward()
    reference(inputs).sum().backward()

    actor._optimizer_step(domain_gradients=None, domain_names=["math"])
    reference_optimizer.step()

    torch.testing.assert_close(module.weight.detach(), reference.weight.detach())
    assert actor._last_mopd_projection_metrics == {}


def test_multi_domain_projection_cannot_silently_skip_missing_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1, weight_decay=0.0)
    config = FSDPActorConfig(
        ppo_micro_batch_size_per_gpu=1,
        grad_clip=0.0,
        use_torch_compile=False,
    )
    actor = DataParallelPPOActor(config=config, actor_module=module, actor_optimizer=optimizer)
    actor._mopd_projection_mode = "adam_project_hard"
    module(torch.tensor([[1.0, -2.0]])).sum().backward()
    before = module.weight.detach().clone()

    with pytest.raises(ValueError, match="replayed gradients"):
        actor._optimizer_step(domain_gradients=None, domain_names=["math", "code"])
    with pytest.raises(ValueError, match="same length"):
        actor._optimizer_step(domain_gradients=[[], []], domain_names=["math", "code", "if"])

    torch.testing.assert_close(module.weight.detach(), before)
    assert not optimizer.state


def test_domain_replays_reconstruct_mixed_surrogate_across_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_id", lambda: torch.device("cpu"))
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    config = FSDPActorConfig(
        ppo_mini_batch_size=4,
        ppo_micro_batch_size_per_gpu=2,
        grad_clip=0.0,
        use_torch_compile=False,
    )
    actor = _ToyProjectionActor(config=config, actor_module=module, actor_optimizer=optimizer)
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 1.0], [1.0, -1.0]],
            [[-1.0, 0.5], [2.0, 1.0], [0.0, 1.0]],
            [[0.25, -0.5], [1.5, 0.5], [-1.0, -1.0]],
            [[2.0, -1.0], [0.5, 0.25], [1.0, 2.0]],
        ]
    )
    response_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    advantages = torch.tensor([[1.0, -0.5, 0.25], [-1.0, 0.0, 0.0], [0.5, 1.5, 0.0], [-0.25, 0.75, 1.0]])
    domain_loss_weight = torch.tensor([1.0, 2.0, 0.5, 1.0])
    domain_ids = torch.tensor([0, 1, 2, 0])
    mini_batch = DataProto.from_dict(
        tensors={
            "responses": torch.zeros(4, 3, dtype=torch.long),
            "response_mask": response_mask,
            "advantages": advantages,
            "mopd_projection_domain_ids": domain_ids,
            "domain_loss_weight": domain_loss_weight,
            "toy_features": features,
        }
    )
    parameters = optimizer_parameters(optimizer)
    domain_gradients = []
    for domain_index in range(3):
        optimizer.zero_grad()
        actor._backward_mopd_projection_domain(mini_batch, domain_index=domain_index, temperature=1.0)
        domain_gradients.append(snapshot_gradients(parameters, offload="device"))

    optimizer.zero_grad()
    for micro_batch in mini_batch.split(2):
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        log_probs = module(model_inputs["toy_features"]).squeeze(-1)
        weighted_advantages = model_inputs["advantages"] * model_inputs["domain_loss_weight"].unsqueeze(-1)
        token_loss = -(weighted_advantages * log_probs)
        loss = (token_loss * model_inputs["response_mask"]).sum() / (model_inputs["response_mask"].sum() + 1e-8)
        (loss / 2).backward()

    _, _, relative_error = gradient_decomposition_metrics(parameters, domain_gradients)
    assert relative_error < 1e-6


def test_lattice_stall_falls_back_to_exact_zero_parameter_delta() -> None:
    before = torch.tensor([-0.0005645627970807254, -0.0017002675449475646], dtype=torch.float32)
    gradient = torch.tensor([7.009509772615274e-07, 15.987810134887695], dtype=torch.float32)
    nominal_delta = torch.tensor([7.77473469497636e-05, 1.924165553646162e-05], dtype=torch.float32)
    parameter = torch.nn.Parameter((before + nominal_delta).clone())

    result = project_parameter_step_(
        [parameter],
        [before.clone()],
        [[gradient]],
        chunk_numel=1,
    )

    assert result.zero_fallback
    assert result.corrective_retries == 2
    assert result.corrective_retries_exhausted
    assert result.committed_harms == (0.0,)
    torch.testing.assert_close(parameter.detach(), before, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("before", "gradient", "raw_delta", "expected_retries"),
    [
        (
            [-0.11239627748727798, 9.010202407836914],
            [-0.10342538356781006, -0.00011842021194752306],
            [-0.004917423240840435, 5.141821020515636e-06],
            1,
        ),
        (
            [-0.005708897486329079, 0.6878484487533569],
            [-1.284565769310575e-07, -0.001323094591498375],
            [-6.268937431741506e-05, -3.154683145112358e-05],
            2,
        ),
    ],
)
def test_guarded_corrective_projection_succeeds_on_fp32_boundary_cases(
    before,
    gradient,
    raw_delta,
    expected_retries,
) -> None:
    before_tensor = torch.tensor(before, dtype=torch.float32)
    gradient_tensor = torch.tensor(gradient, dtype=torch.float32)
    parameter = torch.nn.Parameter((before_tensor + torch.tensor(raw_delta, dtype=torch.float32)).clone())

    result = project_parameter_step_(
        [parameter],
        [before_tensor.clone()],
        [[gradient_tensor]],
        chunk_numel=1,
    )

    committed_delta = parameter.detach().double() - before_tensor.double()
    assert not result.zero_fallback
    assert result.corrective_retries == expected_retries
    assert torch.dot(gradient_tensor.double(), committed_delta).item() <= 0.0


def test_corrective_projection_longer_than_current_update_uses_zero_fallback() -> None:
    before = torch.tensor([-11.2627773, -17.0824299], dtype=torch.float32)
    raw_delta = torch.tensor([-9.5367431640625e-7, 0.0], dtype=torch.float32)
    gradient = torch.tensor([-2.790952e-8, -2.911655e-8], dtype=torch.float32)
    parameter = torch.nn.Parameter((before + raw_delta).clone())

    result = project_parameter_step_(
        [parameter],
        [before.clone()],
        [[gradient]],
        chunk_numel=1,
    )

    assert result.zero_fallback
    assert result.corrective_failed
    assert result.max_corrective_norm_ratio > 1.0
    torch.testing.assert_close(parameter.detach(), before, rtol=0.0, atol=0.0)


def test_zero_fallback_preserves_already_advanced_adamw_state() -> None:
    before = torch.tensor([-11.2627773, -17.0824299], dtype=torch.float32)
    parameter = torch.nn.Parameter(before.clone())
    optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.0)
    parameter.grad = torch.tensor([0.5, -0.25], dtype=torch.float32)
    optimizer.step()
    state_before_projection = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[parameter].items()
    }

    raw_delta = torch.tensor([-9.5367431640625e-7, 0.0], dtype=torch.float32)
    gradient = torch.tensor([-2.790952e-8, -2.911655e-8], dtype=torch.float32)
    with torch.no_grad():
        parameter.copy_(before + raw_delta)
    result = project_parameter_step_(
        [parameter],
        [before.clone()],
        [[gradient]],
        chunk_numel=1,
    )

    assert result.zero_fallback
    torch.testing.assert_close(parameter.detach(), before, rtol=0.0, atol=0.0)
    for key, expected in state_before_projection.items():
        actual = optimizer.state[parameter][key]
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected)
        else:
            assert actual == expected


def test_simulated_shard_statistics_match_full_vector_projection() -> None:
    raw_delta = torch.tensor([0.3, -0.2, 0.1, 0.4], dtype=torch.float64)
    gradients = [
        torch.tensor([1.0, 2.0, -1.0, 0.0], dtype=torch.float64),
        torch.tensor([-2.0, 0.5, 1.0, 1.0], dtype=torch.float64),
        torch.tensor([0.0, -1.0, 0.5, 2.0], dtype=torch.float64),
    ]
    full_gram = torch.stack(gradients) @ torch.stack(gradients).T
    full_harms = torch.stack(gradients) @ raw_delta

    shard_grams = []
    shard_harms = []
    for shard in (slice(0, 2), slice(2, 4)):
        shard_gradients = torch.stack([gradient[shard] for gradient in gradients])
        shard_grams.append(shard_gradients @ shard_gradients.T)
        shard_harms.append(shard_gradients @ raw_delta[shard])

    torch.testing.assert_close(sum(shard_grams), full_gram)
    torch.testing.assert_close(sum(shard_harms), full_harms)
    full = solve_update_projection(full_gram.tolist(), full_harms.tolist())
    reduced = solve_update_projection(sum(shard_grams).tolist(), sum(shard_harms).tolist())
    assert reduced.correction_coefficients == pytest.approx(full.correction_coefficients)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_gloo_projection_matches_full_vector() -> None:
    before = torch.tensor([1.0, -1.0, 2.0, -2.0], dtype=torch.float32)
    raw_delta = torch.tensor([0.3, -0.2, 0.1, 0.4], dtype=torch.float32)
    domain_gradients = [
        torch.tensor([1.0, 2.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([-2.0, 0.5, 1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.5, 2.0], dtype=torch.float32),
    ]
    full_parameter = torch.nn.Parameter((before + raw_delta).clone())
    project_parameter_step_(
        [full_parameter],
        [before.clone()],
        [[gradient] for gradient in domain_gradients],
        chunk_numel=2,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous_socket:
        rendezvous_socket.bind(("127.0.0.1", 0))
        rendezvous_port = rendezvous_socket.getsockname()[1]
    init_method = f"tcp://127.0.0.1:{rendezvous_port}"

    mp.spawn(
        _distributed_projection_worker,
        args=(2, init_method, before, raw_delta, domain_gradients, full_parameter.detach()),
        nprocs=2,
        join=True,
    )
