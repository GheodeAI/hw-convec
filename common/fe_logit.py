"""Binomial logit with absorbed cell-month effects, year effects and one exposure.

Shared by the warm-season FE model (fit_cg_heatwave_fixed_effects.py) and its
delta-method margin report (report_warmseason_fe_margins.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class FitResult:
    alpha: np.ndarray
    year_effects: np.ndarray
    beta: float
    converged: bool
    iterations: int
    log_likelihood: float
    mu: np.ndarray
    weights: np.ndarray


def log_likelihood(successes: np.ndarray, totals: np.ndarray, eta: np.ndarray) -> float:
    mu = np.clip(expit(eta), 1e-15, 1.0 - 1e-15)
    return float(np.sum(successes * np.log(mu) + (totals - successes) * np.log1p(-mu)))


def newton_step(
    group: np.ndarray,
    year: np.ndarray,
    exposure: np.ndarray,
    totals: np.ndarray,
    successes: np.ndarray,
    alpha: np.ndarray,
    year_effects: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    n_groups = alpha.size
    n_years = year_effects.size + 1
    gamma = np.r_[0.0, year_effects]
    eta = alpha[group] + gamma[year] + beta * exposure
    mu = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
    weights = totals * mu * (1.0 - mu)
    score = successes - totals * mu

    group_weight = np.bincount(group, weights=weights, minlength=n_groups)
    group_score = np.bincount(group, weights=score, minlength=n_groups)
    if np.any(group_weight <= 0):
        raise RuntimeError("A retained fixed-effect group has zero IRLS weight")

    group_year_weight = np.bincount(
        group * n_years + year,
        weights=weights,
        minlength=n_groups * n_years,
    ).reshape(n_groups, n_years)
    by = group_year_weight[:, 1:]
    bx = np.bincount(group, weights=weights * exposure, minlength=n_groups)
    design_cross = np.column_stack([by, bx])

    year_weight = np.bincount(year, weights=weights, minlength=n_years)[1:]
    year_x_weight = np.bincount(year, weights=weights * exposure, minlength=n_years)[1:]
    c = np.zeros((n_years, n_years), dtype=np.float64)
    c[:-1, :-1] = np.diag(year_weight)
    c[:-1, -1] = year_x_weight
    c[-1, :-1] = year_x_weight
    c[-1, -1] = np.sum(weights * exposure * exposure)

    score_year = np.bincount(year, weights=score, minlength=n_years)[1:]
    score_z = np.r_[score_year, np.sum(score * exposure)]
    inverse_group_weight = 1.0 / group_weight
    schur = c - design_cross.T @ (design_cross * inverse_group_weight[:, None])
    rhs = score_z - design_cross.T @ (group_score * inverse_group_weight)
    delta_z = np.linalg.solve(schur, rhs)
    delta_alpha = (group_score - design_cross @ delta_z) * inverse_group_weight
    return delta_alpha, delta_z[:-1], float(delta_z[-1]), mu, weights


def fit_absorbed_logit(
    group: np.ndarray,
    year: np.ndarray,
    exposure: np.ndarray,
    totals: np.ndarray,
    successes: np.ndarray,
    n_groups: int,
    n_years: int,
    max_iter: int,
    tolerance: float,
) -> FitResult:
    group_totals = np.bincount(group, weights=totals, minlength=n_groups)
    group_successes = np.bincount(group, weights=successes, minlength=n_groups)
    initial_rate = np.clip(group_successes / group_totals, 1e-6, 1.0 - 1e-6)
    alpha = np.log(initial_rate / (1.0 - initial_rate))
    year_effects = np.zeros(n_years - 1, dtype=np.float64)
    beta = 0.0
    gamma = np.r_[0.0, year_effects]
    eta = alpha[group] + gamma[year] + beta * exposure
    current_ll = log_likelihood(successes, totals, eta)
    converged = False

    for iteration in range(1, max_iter + 1):
        delta_alpha, delta_year, delta_beta, _, _ = newton_step(
            group,
            year,
            exposure,
            totals,
            successes,
            alpha,
            year_effects,
            beta,
        )
        max_step = max(
            float(np.max(np.abs(delta_alpha))),
            float(np.max(np.abs(delta_year))) if delta_year.size else 0.0,
            abs(delta_beta),
        )

        step_scale = 1.0
        accepted = False
        while step_scale >= 2.0**-20:
            proposed_alpha = alpha + step_scale * delta_alpha
            proposed_year = year_effects + step_scale * delta_year
            proposed_beta = beta + step_scale * delta_beta
            proposed_gamma = np.r_[0.0, proposed_year]
            proposed_eta = (
                proposed_alpha[group]
                + proposed_gamma[year]
                + proposed_beta * exposure
            )
            proposed_ll = log_likelihood(successes, totals, proposed_eta)
            if proposed_ll >= current_ll - 1e-7:
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            raise RuntimeError("Logit Newton line search failed")

        ll_change = proposed_ll - current_ll
        alpha = proposed_alpha
        year_effects = proposed_year
        beta = proposed_beta
        current_ll = proposed_ll
        print(
            f"IRLS {iteration:02d}: logLik={current_ll:.6f}, "
            f"beta={beta:.8f}, max_step={max_step * step_scale:.3e}"
        )
        if max_step * step_scale < tolerance or (
            abs(ll_change) < tolerance * (1.0 + abs(current_ll))
            and max_step * step_scale < math.sqrt(tolerance)
        ):
            converged = True
            break

    gamma = np.r_[0.0, year_effects]
    eta = alpha[group] + gamma[year] + beta * exposure
    mu = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
    weights = totals * mu * (1.0 - mu)
    return FitResult(
        alpha=alpha,
        year_effects=year_effects,
        beta=beta,
        converged=converged,
        iterations=iteration,
        log_likelihood=current_ll,
        mu=mu,
        weights=weights,
    )


def residualize_exposure(
    group: np.ndarray,
    year: np.ndarray,
    exposure: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
    n_years: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    group_weight = np.bincount(group, weights=weights, minlength=n_groups)
    group_x_weight = np.bincount(group, weights=weights * exposure, minlength=n_groups)
    group_year_weight = np.bincount(
        group * n_years + year,
        weights=weights,
        minlength=n_groups * n_years,
    ).reshape(n_groups, n_years)
    by = group_year_weight[:, 1:]
    year_weight = np.bincount(year, weights=weights, minlength=n_years)[1:]
    year_x_weight = np.bincount(year, weights=weights * exposure, minlength=n_years)[1:]
    inverse_group_weight = 1.0 / group_weight
    lhs = np.diag(year_weight) - by.T @ (by * inverse_group_weight[:, None])
    rhs = year_x_weight - by.T @ (group_x_weight * inverse_group_weight)
    year_projection = np.linalg.solve(lhs, rhs)
    group_projection = (group_x_weight - by @ year_projection) * inverse_group_weight
    gamma_projection = np.r_[0.0, year_projection]
    residual = exposure - group_projection[group] - gamma_projection[year]
    bread = float(np.sum(weights * residual * residual))
    return residual, group_projection, year_projection, bread
