# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UP-MOPD: update projection for multi-teacher on-policy distillation.

This module solves the projection dual for the active teacher domains and commits
the projected *parameter displacement* after the optimizer has advanced its state.

The integration intentionally uses the displacement produced by the actual optimizer
instead of reimplementing AdamW.  This keeps the contract exact for the configured
PyTorch optimizer: moments and the optimizer step advance on the ordinary mixed
gradient, while the parameter is replaced by the closest first-order-safe target.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

__all__ = [
    "ProjectionNumericalError",
    "GuardedProjectionCorrection",
    "UpdateProjection",
    "ParameterStepProjection",
    "gradient_decomposition_metrics",
    "materialize_projected_parameter_target",
    "optimizer_parameters",
    "project_parameter_step_",
    "snapshot_gradients",
    "snapshot_parameters",
    "solve_guarded_update_correction",
    "solve_update_projection",
]


class ProjectionNumericalError(ValueError):
    """Raised when a strict hard projection is numerically unresolved."""


@dataclass(frozen=True)
class UpdateProjection:
    """Euclidean projection of one optimizer displacement onto domain half-spaces."""

    correction_coefficients: tuple[float, ...]
    dual_variables: tuple[float, ...]
    projected_harms: tuple[float, ...]
    active_set: tuple[int, ...]


@dataclass(frozen=True)
class ProjectedTargetMaterialization:
    """A projected target and the displacement realized by the storage dtype."""

    target: torch.Tensor
    reference_delta: torch.Tensor
    materialized_delta: torch.Tensor


@dataclass(frozen=True)
class ProjectionMaterialization:
    """Global diagnostics for one materialized projection attempt."""

    reference_harms: tuple[float, ...]
    materialized_harms: tuple[float, ...]
    reference_delta_norm: float
    materialized_delta_norm: float
    rounding_error_norm: float


@dataclass(frozen=True)
class GuardedProjectionCorrection:
    """One scale-normalized interior correction for a materialized projection."""

    projection: UpdateProjection
    guard_distance: float
    correction_norm_squared: float


@dataclass(frozen=True)
class ParameterStepProjection:
    """Result of committing one strict projected optimizer step."""

    correction_coefficients: tuple[float, ...]
    gradient_gram: tuple[tuple[float, ...], ...]
    gradient_norms: tuple[float, ...]
    active_set: tuple[int, ...]
    nominal_harms: tuple[float, ...]
    analytic_projected_harms: tuple[float, ...]
    first_attempt: ProjectionMaterialization
    final_attempt: ProjectionMaterialization
    committed_harms: tuple[float, ...]
    corrective_retries: int
    max_corrective_guard_distance: float
    max_corrective_norm_ratio: float
    solver_failed: bool
    corrective_failed: bool
    corrective_retries_exhausted: bool
    final_recheck_failed: bool
    zero_fallback: bool
    attempted_normalized_violation: float
    final_normalized_violation: float


def optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    """Return optimizer parameters once, in stable param-group order."""

    parameters: list[torch.Tensor] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
    return parameters


@torch.no_grad()
def snapshot_parameters(parameters: Sequence[torch.Tensor], offload: str = "cpu") -> list[torch.Tensor]:
    """Copy optimizer parameters before a candidate step."""

    snapshots = []
    for parameter in parameters:
        if offload == "cpu":
            snapshots.append(parameter.detach().to(device="cpu", copy=True))
        elif offload == "device":
            snapshots.append(parameter.detach().to(device=parameter.device, copy=True))
        else:
            raise ValueError(f"Unknown projection offload target: {offload!r}.")
    return snapshots


@torch.no_grad()
def snapshot_gradients(parameters: Sequence[torch.Tensor], offload: str = "cpu") -> list[torch.Tensor | None]:
    """Copy one sharded domain-gradient contribution in FP32."""

    snapshots: list[torch.Tensor | None] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            snapshots.append(None)
        elif offload == "cpu":
            snapshots.append(gradient.detach().to(device="cpu", dtype=torch.float32, copy=True))
        elif offload == "device":
            snapshots.append(gradient.detach().to(device=gradient.device, dtype=torch.float32, copy=True))
        else:
            raise ValueError(f"Unknown projection offload target: {offload!r}.")
    return snapshots


def materialize_projected_parameter_target(
    parameter: torch.Tensor,
    nominal_delta: torch.Tensor,
    domain_gradients: Sequence[torch.Tensor],
    correction_coefficients: Sequence[float],
) -> ProjectedTargetMaterialization:
    """Form the projected delta in FP64, then round the absolute target once."""

    if len(domain_gradients) != len(correction_coefficients):
        raise ValueError("domain_gradients and correction_coefficients must have the same length.")
    if parameter.shape != nominal_delta.shape:
        raise ValueError("parameter and nominal_delta must have the same shape.")
    if not parameter.is_floating_point() or not nominal_delta.is_floating_point():
        raise TypeError("parameter and nominal_delta must be floating-point tensors.")
    if parameter.device != nominal_delta.device:
        raise ValueError("parameter and nominal_delta must be on the same device.")

    reference_delta = nominal_delta.to(dtype=torch.float64, copy=True)
    for gradient, coefficient in zip(domain_gradients, correction_coefficients, strict=True):
        if gradient.shape != parameter.shape:
            raise ValueError("Every domain gradient must have the same shape as parameter.")
        if gradient.device != parameter.device:
            raise ValueError("Every domain gradient must be on the same device as parameter.")
        coefficient = float(coefficient)
        if not math.isfinite(coefficient):
            raise ValueError("Projection correction coefficients must be finite.")
        reference_delta.add_(gradient.double(), alpha=-coefficient)

    parameter_fp64 = parameter.double()
    target = (parameter_fp64 + reference_delta).to(dtype=parameter.dtype)
    materialized_delta = target.double() - parameter_fp64
    return ProjectedTargetMaterialization(
        target=target,
        reference_delta=reference_delta,
        materialized_delta=materialized_delta,
    )


def _validate_projection_inputs(
    gradient_gram: Sequence[Sequence[float]], nominal_harms: Sequence[float], epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    n_domains = len(nominal_harms)
    if n_domains == 0:
        raise ValueError("At least one domain constraint is required.")
    if n_domains > 8:
        raise ValueError("Active-set projection supports at most eight domain constraints.")
    if len(gradient_gram) != n_domains or any(len(row) != n_domains for row in gradient_gram):
        raise ValueError("gradient_gram must be square and match nominal_harms.")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative.")

    gram = torch.tensor(gradient_gram, dtype=torch.float64)
    harms = torch.tensor(nominal_harms, dtype=torch.float64)
    if not bool(torch.isfinite(gram).all()) or not bool(torch.isfinite(harms).all()):
        raise ValueError("Projection statistics must contain only finite values.")
    scale = max(1.0, float(gram.abs().max().item()))
    if not torch.allclose(gram, gram.T, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError("gradient_gram must be symmetric.")
    gram = 0.5 * (gram + gram.T)
    eigenvalues = torch.linalg.eigvalsh(gram)
    if float(eigenvalues.min().item()) < -1e-9 * scale:
        raise ValueError("gradient_gram must be positive semidefinite.")
    return gram, harms


def _solve_nonnegative_quadratic(matrix: torch.Tensor, linear: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Solve ``min_{x>=0} 0.5 x^T A x - b^T x`` by active-set enumeration."""

    n = int(linear.numel())
    scale = max(1.0, float(matrix.abs().max().item()), float(linear.abs().max().item()))
    tolerance = 2e-9 * scale
    candidates: list[tuple[float, tuple[int, ...], torch.Tensor]] = []

    for active_size in range(n + 1):
        for active in itertools.combinations(range(n), active_size):
            solution = torch.zeros(n, dtype=torch.float64)
            if active:
                index = torch.tensor(active, dtype=torch.long)
                submatrix = matrix.index_select(0, index).index_select(1, index)
                rhs = linear.index_select(0, index)
                eigenvalues = torch.linalg.eigvalsh(submatrix)
                largest_eigenvalue = float(eigenvalues[-1].item())
                smallest_eigenvalue = float(eigenvalues[0].item())
                use_direct_solve = smallest_eigenvalue > 0.0
                if use_direct_solve and len(active) > 1:
                    relative_eigenvalue = smallest_eigenvalue / max(largest_eigenvalue, 1e-300)
                    if relative_eigenvalue < 1e-10:
                        continue
                if use_direct_solve:
                    try:
                        active_solution = torch.linalg.solve(submatrix, rhs)
                    except RuntimeError:
                        continue
                else:
                    try:
                        active_solution = torch.linalg.lstsq(submatrix, rhs).solution
                    except RuntimeError:
                        continue
                invalid_negative = any(
                    float(value.item()) < -1e-12 * max(1.0, abs(float(value.item()))) for value in active_solution
                )
                if invalid_negative:
                    continue
                solution[index] = active_solution.clamp_min(0.0)

            gradient = matrix @ solution - linear
            active_set = set(active)
            valid = True
            for idx in range(n):
                component_scale = float(torch.sum((matrix[idx] * solution).abs()).item()) + abs(
                    float(linear[idx].item())
                )
                component_tolerance = max(
                    tolerance,
                    1e-10 * component_scale,
                    64.0 * math.ulp(component_scale),
                )
                if idx in active_set:
                    if abs(float(gradient[idx].item())) > component_tolerance:
                        valid = False
                        break
                elif float(gradient[idx].item()) < -component_tolerance:
                    valid = False
                    break
            if not valid:
                continue

            objective = float((0.5 * solution @ matrix @ solution - linear @ solution).item())
            candidates.append((objective, active, solution))

    if not candidates:
        raise ProjectionNumericalError("Unable to solve the non-negative projection dual.")
    _, active, solution = min(candidates, key=lambda item: (item[0], len(item[1]), item[1]))
    return solution, active


def solve_update_projection(
    gradient_gram: Sequence[Sequence[float]],
    nominal_harms: Sequence[float],
    *,
    epsilon: float = 0.0,
) -> UpdateProjection:
    """Project one parameter displacement against one or more domain gradients."""

    gram, harms = _validate_projection_inputs(gradient_gram, nominal_harms, epsilon)
    norms = torch.sqrt(torch.diagonal(gram).clamp_min(0.0))
    nonzero = norms > 0
    if bool(((~nonzero) & (harms != 0)).any()):
        raise ValueError("A zero domain gradient cannot have non-zero nominal harm.")
    safe_norms = torch.where(nonzero, norms, torch.ones_like(norms))
    normalized_gram = gram / safe_norms[:, None] / safe_norms[None, :]
    residual = (harms - epsilon) / safe_norms
    residual = torch.where(nonzero, residual, torch.zeros_like(residual))

    dual, active_set = _solve_nonnegative_quadratic(normalized_gram, residual)
    coefficients = dual / safe_norms
    projected_harms = harms - gram @ coefficients
    result_scale = max(
        1.0,
        float(harms.abs().max().item()),
        float(projected_harms.abs().max().item()),
        abs(float(epsilon)),
    )
    if bool((projected_harms > epsilon + 1e-8 * result_scale).any()):
        raise ProjectionNumericalError("The projected update failed its analytic constraint check.")

    return UpdateProjection(
        correction_coefficients=tuple(float(value) for value in coefficients.tolist()),
        dual_variables=tuple(float(value) for value in dual.tolist()),
        projected_harms=tuple(float(value) for value in projected_harms.tolist()),
        active_set=tuple(active_set),
    )


def solve_guarded_update_correction(
    gradient_gram: Sequence[Sequence[float]],
    materialized_harms: Sequence[float],
    *,
    epsilon: float = 0.0,
    guard_multiplier: float,
) -> GuardedProjectionCorrection:
    """Push storage-dtype residuals inside all hard half-spaces."""

    gram, harms = _validate_projection_inputs(gradient_gram, materialized_harms, epsilon)
    if not math.isfinite(guard_multiplier) or guard_multiplier < 0.0:
        raise ValueError("guard_multiplier must be finite and non-negative.")
    norms = torch.sqrt(torch.diagonal(gram).clamp_min(0.0))
    residual_distances = []
    for harm, norm in zip(harms.tolist(), norms.tolist(), strict=True):
        if norm == 0.0:
            if harm != 0.0:
                raise ProjectionNumericalError("A zero domain gradient cannot have non-zero materialized harm.")
            residual_distances.append(0.0)
        else:
            residual_distances.append(max(0.0, float(harm) - epsilon) / float(norm))
    guard_distance = guard_multiplier * max(residual_distances, default=0.0)
    guarded_harms = tuple(
        float(harm) + guard_distance * float(norm) for harm, norm in zip(harms.tolist(), norms.tolist(), strict=True)
    )
    projection = solve_update_projection(gram.tolist(), guarded_harms, epsilon=epsilon)
    coefficients = torch.tensor(projection.correction_coefficients, dtype=torch.float64)
    correction_norm_squared = float((coefficients @ gram @ coefficients).item())
    return GuardedProjectionCorrection(
        projection=projection,
        guard_distance=guard_distance,
        correction_norm_squared=correction_norm_squared,
    )


def _all_reduce_sum_(tensor: torch.Tensor, process_group=None) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)


def _chunk_on_device(
    tensor: torch.Tensor | None,
    *,
    start: int,
    end: int,
    device: torch.device,
    like: torch.Tensor,
) -> torch.Tensor:
    if tensor is None:
        return torch.zeros_like(like, dtype=torch.float32, device=device)
    return tensor.reshape(-1)[start:end].to(device=device, dtype=torch.float32, non_blocking=False)


@torch.no_grad()
def gradient_decomposition_metrics(
    parameters: Sequence[torch.Tensor],
    domain_gradients: Sequence[Sequence[torch.Tensor | None]],
    *,
    chunk_numel: int = 1_000_000,
    process_group=None,
) -> tuple[float, float, float]:
    """Compare the ordinary mixed gradient with the sum of domain contributions."""

    if not domain_gradients:
        raise ValueError("At least one domain gradient is required.")
    if any(len(gradients) != len(parameters) for gradients in domain_gradients):
        raise ValueError("Every domain gradient list must align with parameters.")
    device = next((parameter.device for parameter in parameters if parameter.numel()), torch.device("cpu"))
    statistics = torch.zeros(2, dtype=torch.float64, device=device)
    for parameter_index, parameter in enumerate(parameters):
        raw_gradient = parameter.grad
        if raw_gradient is None and all(gradients[parameter_index] is None for gradients in domain_gradients):
            continue
        raw_flat = None if raw_gradient is None else raw_gradient.detach().reshape(-1)
        for start in range(0, parameter.numel(), chunk_numel):
            end = min(start + chunk_numel, parameter.numel())
            if raw_flat is None:
                raw_chunk = torch.zeros(end - start, dtype=torch.float32, device=device)
            else:
                raw_chunk = raw_flat[start:end].to(device=device, dtype=torch.float32)
            reconstructed = torch.zeros_like(raw_chunk)
            for gradients in domain_gradients:
                reconstructed.add_(
                    _chunk_on_device(gradients[parameter_index], start=start, end=end, device=device, like=raw_chunk)
                )
            error = reconstructed.double() - raw_chunk.double()
            statistics[0] += torch.sum(error.square())
            statistics[1] += torch.sum(raw_chunk.double().square())
    _all_reduce_sum_(statistics, process_group)
    error_norm = math.sqrt(max(float(statistics[0].item()), 0.0))
    raw_norm = math.sqrt(max(float(statistics[1].item()), 0.0))
    relative_error = error_norm / max(raw_norm, 1e-30)
    return error_norm, raw_norm, relative_error


def _projection_statistics(
    parameters: Sequence[torch.Tensor],
    parameters_before: Sequence[torch.Tensor],
    domain_gradients: Sequence[Sequence[torch.Tensor | None]],
    *,
    chunk_numel: int,
    process_group=None,
) -> tuple[list[list[float]], list[float]]:
    n_domains = len(domain_gradients)
    device = next((parameter.device for parameter in parameters if parameter.numel()), torch.device("cpu"))
    statistics = torch.zeros((n_domains, n_domains + 1), dtype=torch.float64, device=device)
    for parameter_index, (parameter, parameter_before) in enumerate(zip(parameters, parameters_before, strict=True)):
        current_flat = parameter.detach().reshape(-1)
        before_flat = parameter_before.reshape(-1)
        for start in range(0, parameter.numel(), chunk_numel):
            end = min(start + chunk_numel, parameter.numel())
            before_chunk = before_flat[start:end].to(device=device, dtype=torch.float64, non_blocking=False)
            nominal_delta = current_flat[start:end].to(dtype=torch.float64) - before_chunk
            gradient_chunks = [
                _chunk_on_device(
                    gradients[parameter_index],
                    start=start,
                    end=end,
                    device=device,
                    like=current_flat[start:end],
                ).double()
                for gradients in domain_gradients
            ]
            for i, gradient_i in enumerate(gradient_chunks):
                statistics[i, -1] += torch.sum(gradient_i * nominal_delta)
                for j in range(i, n_domains):
                    value = torch.sum(gradient_i * gradient_chunks[j])
                    statistics[i, j] += value
                    if i != j:
                        statistics[j, i] += value
    _all_reduce_sum_(statistics, process_group)
    if not bool(torch.isfinite(statistics).all()):
        raise FloatingPointError("Non-finite projection Gram or nominal harm.")
    return statistics[:, :n_domains].cpu().tolist(), statistics[:, -1].cpu().tolist()


def _probe_materialization(
    parameters: Sequence[torch.Tensor],
    parameters_before: Sequence[torch.Tensor],
    domain_gradients: Sequence[Sequence[torch.Tensor | None]],
    correction_coefficients: Sequence[float],
    *,
    chunk_numel: int,
    write: bool,
    process_group=None,
) -> ProjectionMaterialization:
    n_domains = len(domain_gradients)
    device = next((parameter.device for parameter in parameters if parameter.numel()), torch.device("cpu"))
    statistics = torch.zeros(2 * n_domains + 3, dtype=torch.float64, device=device)
    targets_finite = torch.ones(1, dtype=torch.int32, device=device)

    for parameter_index, (parameter, parameter_before) in enumerate(zip(parameters, parameters_before, strict=True)):
        current_flat = parameter.detach().reshape(-1)
        before_flat = parameter_before.reshape(-1)
        for start in range(0, parameter.numel(), chunk_numel):
            end = min(start + chunk_numel, parameter.numel())
            before_chunk = before_flat[start:end].to(device=device, dtype=parameter.dtype, non_blocking=False)
            nominal_delta = current_flat[start:end].to(dtype=torch.float64) - before_chunk.double()
            gradient_chunks = [
                _chunk_on_device(
                    gradients[parameter_index],
                    start=start,
                    end=end,
                    device=device,
                    like=current_flat[start:end],
                )
                for gradients in domain_gradients
            ]
            materialization = materialize_projected_parameter_target(
                before_chunk,
                nominal_delta,
                gradient_chunks,
                correction_coefficients,
            )
            if not bool(torch.isfinite(materialization.target).all()):
                targets_finite.zero_()
            rounding_error = materialization.materialized_delta - materialization.reference_delta
            for index, gradient in enumerate(gradient_chunks):
                gradient_fp64 = gradient.double()
                statistics[index] += torch.sum(gradient_fp64 * materialization.reference_delta)
                statistics[n_domains + index] += torch.sum(gradient_fp64 * materialization.materialized_delta)
            statistics[-3] += torch.sum(materialization.reference_delta.square())
            statistics[-2] += torch.sum(materialization.materialized_delta.square())
            statistics[-1] += torch.sum(rounding_error.square())
            if write:
                current_flat[start:end].copy_(materialization.target.to(dtype=parameter.dtype))

    _all_reduce_sum_(statistics, process_group)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(targets_finite, op=dist.ReduceOp.MIN, group=process_group)
    if not bool(targets_finite.item()) or not bool(torch.isfinite(statistics).all()):
        raise FloatingPointError("Non-finite projected parameter target.")
    values = statistics.cpu().tolist()
    return ProjectionMaterialization(
        reference_harms=tuple(float(value) for value in values[:n_domains]),
        materialized_harms=tuple(float(value) for value in values[n_domains : 2 * n_domains]),
        reference_delta_norm=math.sqrt(max(float(values[-3]), 0.0)),
        materialized_delta_norm=math.sqrt(max(float(values[-2]), 0.0)),
        rounding_error_norm=math.sqrt(max(float(values[-1]), 0.0)),
    )


@torch.no_grad()
def _restore_parameters(
    parameters: Sequence[torch.Tensor],
    parameters_before: Sequence[torch.Tensor],
    *,
    chunk_numel: int,
) -> None:
    for parameter, before in zip(parameters, parameters_before, strict=True):
        parameter_flat = parameter.reshape(-1)
        before_flat = before.reshape(-1)
        for start in range(0, parameter.numel(), chunk_numel):
            end = min(start + chunk_numel, parameter.numel())
            parameter_flat[start:end].copy_(
                before_flat[start:end].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                    non_blocking=False,
                )
            )


@torch.no_grad()
def project_parameter_step_(
    parameters: Sequence[torch.Tensor],
    parameters_before: Sequence[torch.Tensor],
    domain_gradients: Sequence[Sequence[torch.Tensor | None]],
    *,
    epsilon: float = 0.0,
    guard_multipliers: Sequence[float] = (2.0, 4.0),
    chunk_numel: int = 1_000_000,
    process_group=None,
) -> ParameterStepProjection:
    """Replace an already-applied optimizer step with its strict projected target.

    The caller must snapshot parameters, run the ordinary optimizer exactly once, and
    then call this function before those parameters are used by another forward pass.
    Optimizer state is deliberately left untouched.
    """

    if len(parameters) != len(parameters_before):
        raise ValueError("parameters_before must align with parameters.")
    if not domain_gradients or any(len(gradients) != len(parameters) for gradients in domain_gradients):
        raise ValueError("Every domain gradient list must align with parameters.")
    if chunk_numel <= 0:
        raise ValueError("chunk_numel must be positive.")

    gradient_gram, nominal_harms = _projection_statistics(
        parameters,
        parameters_before,
        domain_gradients,
        chunk_numel=chunk_numel,
        process_group=process_group,
    )
    n_domains = len(domain_gradients)
    gradient_gram_tuple = tuple(tuple(float(value) for value in row) for row in gradient_gram)
    gradient_norms = tuple(math.sqrt(max(float(gradient_gram[index][index]), 0.0)) for index in range(n_domains))

    def normalized_violation(harms: Sequence[float]) -> float:
        return max(
            (max(0.0, float(harm) - epsilon) / max(gradient_norms[index], 1e-30) for index, harm in enumerate(harms)),
            default=0.0,
        )

    zero_materialization = ProjectionMaterialization(
        reference_harms=tuple(0.0 for _ in range(n_domains)),
        materialized_harms=tuple(0.0 for _ in range(n_domains)),
        reference_delta_norm=0.0,
        materialized_delta_norm=0.0,
        rounding_error_norm=0.0,
    )
    try:
        projection = solve_update_projection(gradient_gram, nominal_harms, epsilon=epsilon)
    except ProjectionNumericalError:
        _restore_parameters(parameters, parameters_before, chunk_numel=chunk_numel)
        return ParameterStepProjection(
            correction_coefficients=tuple(0.0 for _ in range(n_domains)),
            gradient_gram=gradient_gram_tuple,
            gradient_norms=gradient_norms,
            active_set=(),
            nominal_harms=tuple(float(value) for value in nominal_harms),
            analytic_projected_harms=tuple(0.0 for _ in range(n_domains)),
            first_attempt=zero_materialization,
            final_attempt=zero_materialization,
            committed_harms=tuple(0.0 for _ in range(n_domains)),
            corrective_retries=0,
            max_corrective_guard_distance=0.0,
            max_corrective_norm_ratio=0.0,
            solver_failed=True,
            corrective_failed=False,
            corrective_retries_exhausted=False,
            final_recheck_failed=False,
            zero_fallback=True,
            attempted_normalized_violation=0.0,
            final_normalized_violation=0.0,
        )

    coefficients = list(projection.correction_coefficients)
    first_attempt = _probe_materialization(
        parameters,
        parameters_before,
        domain_gradients,
        coefficients,
        chunk_numel=chunk_numel,
        write=False,
        process_group=process_group,
    )
    final_attempt = first_attempt
    corrective_retries = 0
    corrective_failed = False
    corrective_retries_exhausted = False
    final_recheck_failed = False
    max_corrective_guard_distance = 0.0
    max_corrective_norm_ratio = 0.0

    for attempt_index, guard_multiplier in enumerate(guard_multipliers):
        if all(harm <= epsilon for harm in final_attempt.materialized_harms):
            break
        try:
            guarded_correction = solve_guarded_update_correction(
                gradient_gram,
                final_attempt.materialized_harms,
                epsilon=epsilon,
                guard_multiplier=float(guard_multiplier),
            )
        except ProjectionNumericalError:
            corrective_failed = True
            break
        max_corrective_guard_distance = max(
            max_corrective_guard_distance,
            guarded_correction.guard_distance,
        )
        extra_norm_squared = guarded_correction.correction_norm_squared
        materialized_norm_squared = final_attempt.materialized_delta_norm**2
        norm_tolerance = 1e-10 * max(materialized_norm_squared, 1e-30)
        if math.isfinite(extra_norm_squared) and extra_norm_squared >= 0.0:
            corrective_norm_ratio = math.sqrt(extra_norm_squared) / max(
                final_attempt.materialized_delta_norm,
                1e-30,
            )
            max_corrective_norm_ratio = max(max_corrective_norm_ratio, corrective_norm_ratio)
        if (
            not math.isfinite(extra_norm_squared)
            or extra_norm_squared < -norm_tolerance
            or extra_norm_squared > materialized_norm_squared + norm_tolerance
        ):
            corrective_failed = True
            break
        coefficients = [
            current + extra
            for current, extra in zip(
                coefficients,
                guarded_correction.projection.correction_coefficients,
                strict=True,
            )
        ]
        if not all(math.isfinite(coefficient) for coefficient in coefficients):
            corrective_failed = True
            break
        corrective_retries += 1
        final_attempt = _probe_materialization(
            parameters,
            parameters_before,
            domain_gradients,
            coefficients,
            chunk_numel=chunk_numel,
            write=False,
            process_group=process_group,
        )
        if attempt_index == len(guard_multipliers) - 1 and not all(
            harm <= epsilon for harm in final_attempt.materialized_harms
        ):
            corrective_retries_exhausted = True

    strict_feasible = all(harm <= epsilon for harm in final_attempt.materialized_harms)
    if not strict_feasible:
        _restore_parameters(parameters, parameters_before, chunk_numel=chunk_numel)
        committed_harms = tuple(0.0 for _ in range(n_domains))
    else:
        committed = _probe_materialization(
            parameters,
            parameters_before,
            domain_gradients,
            coefficients,
            chunk_numel=chunk_numel,
            write=True,
            process_group=process_group,
        )
        if not all(harm <= epsilon for harm in committed.materialized_harms):
            _restore_parameters(parameters, parameters_before, chunk_numel=chunk_numel)
            strict_feasible = False
            final_attempt = committed
            committed_harms = tuple(0.0 for _ in range(n_domains))
            final_recheck_failed = True
        else:
            final_attempt = committed
            committed_harms = committed.materialized_harms

    return ParameterStepProjection(
        correction_coefficients=tuple(float(value) for value in coefficients),
        gradient_gram=gradient_gram_tuple,
        gradient_norms=gradient_norms,
        active_set=projection.active_set,
        nominal_harms=tuple(float(value) for value in nominal_harms),
        analytic_projected_harms=projection.projected_harms,
        first_attempt=first_attempt,
        final_attempt=final_attempt,
        committed_harms=committed_harms,
        corrective_retries=corrective_retries,
        max_corrective_guard_distance=max_corrective_guard_distance,
        max_corrective_norm_ratio=max_corrective_norm_ratio,
        solver_failed=False,
        corrective_failed=corrective_failed,
        corrective_retries_exhausted=corrective_retries_exhausted,
        final_recheck_failed=final_recheck_failed,
        zero_fallback=not strict_feasible,
        attempted_normalized_violation=normalized_violation(first_attempt.materialized_harms),
        final_normalized_violation=normalized_violation(final_attempt.materialized_harms),
    )
