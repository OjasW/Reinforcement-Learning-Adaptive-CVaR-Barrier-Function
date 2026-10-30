
"""Differentiable deterministic CBF slack-QP (BarrierNet style) for diff_cbf_* models.  # [CBF]
 
Same QP as model/jax_cvar_qp.py (objective, per-obstacle slack, box bounds, solver),
but each constraint row is a plain CBF with a learned per-obstacle gain kappa_i:
 
    h_i     = ||d_i||^2 - R_i^2,         d_i = q - p_i   (q = lookahead point, env rel = robot - human)
    hdot_i  = Lg h_i u - 2 d_i . v_i     (constant-velocity obstacle)
    hdot_i + kappa_i h_i >= -s_i
 => -Lg h_i u - s_i <= kappa_i h_i - 2 d_i . v_i
 
Compared with CVaR-BF: one velocity instead of three GMM means (no logsumexp),
no sigma_f * c(beta) tightening, and a learned kappa_i instead of the fixed alpha.
"""


from __future__ import annotations

import math

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy.special as jsp

from solver.jax_qpth import QPFunction


def _extract_blocks(obs):
    batch = obs.shape[0]
    if obs.shape[1] <= 6:
        rel = jnp.zeros((batch, 1, 2), dtype=obs.dtype)
        vel = jnp.zeros_like(rel)
        radius = jnp.zeros((batch, 1), dtype=obs.dtype)
        mask = jnp.zeros((batch, 1), dtype=obs.dtype)
        return rel, vel, radius, mask
    blocks = obs[:, 6:].reshape((batch, -1, 6))
    return blocks[:, :, 0:2], blocks[:, :, 2:4], blocks[:, :, 4], jnp.clip(blocks[:, :, 5], 0.0, 1.0)

def _masked_positive_stats(values, mask):
    values = jnp.where(mask > 0.5, jnp.maximum(values, 0.0), 0.0)
    denom = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    return jnp.sum(values, axis=1) / denom, jnp.max(values, axis=1)


def _masked_raw_stats(values, mask):
    values = jnp.where(mask > 0.5, values, 0.0)
    denom = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    return jnp.sum(values, axis=1) / denom, jnp.max(values, axis=1)


def _single_integrator_qp_objective(hparams, obs, nominal_xy):
    batch = obs.shape[0]
    eye = jnp.eye(hparams.act_dim, dtype=obs.dtype)[None, :, :]
    q = 2.0 * jnp.broadcast_to(eye, (batch, hparams.act_dim, hparams.act_dim))
    p = -2.0 * nominal_xy
    return q, p

def _solve_slack_qp(params, hparams, obs, q, p, g, h_qp):
    batch = obs.shape[0]
    dtype = obs.dtype
    slack_dim = g.shape[1]
    slack_eye = jnp.eye(slack_dim, dtype=dtype)[None, :, :]
    slack_eye = jnp.broadcast_to(slack_eye, (batch, slack_dim, slack_dim))
    zeros_u_slack = jnp.zeros((batch, hparams.act_dim, slack_dim), dtype=dtype)
    zeros_slack_u = jnp.zeros((batch, slack_dim, hparams.act_dim), dtype=dtype)

    slack_q = 2.0 * hparams.qp_slack_weight * slack_eye
    q_top = jnp.concatenate([q, zeros_u_slack], axis=2)
    q_bottom = jnp.concatenate([zeros_slack_u, slack_q], axis=2)
    q_qp = jnp.concatenate([q_top, q_bottom], axis=1)
    p_qp = jnp.concatenate([p, jnp.zeros((batch, slack_dim), dtype=dtype)], axis=1)

    cbf_g = jnp.concatenate([g, -slack_eye], axis=2)
    slack_g = jnp.concatenate([zeros_slack_u, -slack_eye], axis=2)
    g_qp = jnp.concatenate([cbf_g, slack_g], axis=1)
    h_qp_full = jnp.concatenate([h_qp, jnp.zeros((batch, slack_dim), dtype=dtype)], axis=1)

    total_dim = q_qp.shape[1]
    bound_eye = jnp.eye(hparams.act_dim, dtype=dtype)
    bound_pad = jnp.zeros((hparams.act_dim, total_dim - hparams.act_dim), dtype=dtype)
    upper_g = jnp.concatenate([bound_eye, bound_pad], axis=1)
    lower_g = jnp.concatenate([-bound_eye, bound_pad], axis=1)
    bound_g = jnp.concatenate([upper_g, lower_g], axis=0)
    bound_g = jnp.broadcast_to(bound_g[None, :, :], (batch, 2 * hparams.act_dim, total_dim))

    action_low = jnp.asarray(params["action_low"], dtype=dtype)
    action_high = jnp.asarray(params["action_high"], dtype=dtype)
    bound_h = jnp.concatenate([action_high, -action_low], axis=0)
    bound_h = jnp.broadcast_to(bound_h[None, :], (batch, 2 * hparams.act_dim))
    g_qp = jnp.concatenate([g_qp, bound_g], axis=1)
    h_qp_full = jnp.concatenate([h_qp_full, bound_h], axis=1)

    qp_dtype = jnp.float64
    empty = jnp.empty((0,), dtype=qp_dtype)
    solution = QPFunction(
        eps=hparams.qp_eps,
        verbose=hparams.qp_verbose,
        notImprovedLim=hparams.qp_not_improved_lim,
        maxIter=hparams.qp_max_iter,
        check_Q_spd=hparams.qp_check_q_spd,
    )(
        q_qp.astype(qp_dtype),
        p_qp.astype(qp_dtype),
        g_qp.astype(qp_dtype),
        h_qp_full.astype(qp_dtype),
        empty,
        empty,
    )
    action = solution[:, : hparams.act_dim].astype(dtype)
    slack = solution[:, hparams.act_dim : hparams.act_dim + slack_dim].astype(dtype)
    return action, jnp.maximum(slack, 0.0)


def _unicycle_qp_objective(hparams, obs, nominal_xy):
    dtype = obs.dtype
    theta = obs[:, 4]
    eps = hparams.lookahead_distance
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    j_mat = jnp.stack(
        [
            jnp.stack([cos_t, -eps * sin_t], axis=1),
            jnp.stack([sin_t, eps * cos_t], axis=1),
        ],
        axis=1,
    )
    jt = jnp.swapaxes(j_mat, 1, 2)
    eye = jnp.eye(hparams.act_dim, dtype=dtype)[None, :, :]
    q = 2.0 * jnp.matmul(jt, j_mat) + 1e-6 * eye
    p = -2.0 * jnp.matmul(jt, nominal_xy[:, :, None]).squeeze(-1)
    return q, p


def _per_obstacle(values, obs):
    """Accept (B, ) or (B, K) and return (B, K)-broadcastable values in obs dtype."""
    values = jnp.asarray(values, dtype=obs.dtype)
    if values.ndim == 1:
        values = values[:, None]
    return values

def _unicycle_cbf_constraints(hparams, obs, kappa, r_safe):
    theta = obs[:, 4]
    rel, human_vel, _obstacle_radius, mask = _extract_blocks(obs)

    eps = hparams.lookahead_distance
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    heading = jnp.stack([cos_t, sin_t], axis=1)[:, None, :]
    lookahead_rel = rel + eps * heading

    r_safe = _per_obstacle(r_safe, obs) + eps
    kappa = _per_obstacle(kappa, obs)

    h = jnp.sum(lookahead_rel * lookahead_rel, axis=2) - r_safe * r_safe

    lg_v = 2.0 * (lookahead_rel[:, :, 0] * cos_t[:, None] + lookahead_rel[:, :, 1] * sin_t[:, None])
    lg_w = 2.0 * eps * (lookahead_rel[:, :, 1] * cos_t[:, None] - lookahead_rel[:, :, 0] * sin_t[:, None])

    drift = 2.0 * jnp.sum(lookahead_rel * human_vel, axis=2)
    
    g = jnp.stack([-(lg_v * mask), -(lg_w * mask)], axis=2)
    h_qp = (kappa * h - drift) * mask
    return g, h_qp, mask


def _single_integrator_cbf_constraints(hparams, obs, kappa, r_safe):
    del hparams

    rel, human_vel, _obstacle_radius, mask = _extract_blocks(obs)

    r_safe = _per_obstacle(r_safe, obs)
    kappa = _per_obstacle(kappa, obs)
    h = jnp.sum(rel * rel, axis=2) - r_safe * r_safe
    drift = 2.0 * jnp.sum(rel * human_vel, axis=2)
        
    g = -2.0 * rel * mask[:, :, None]
    h_qp = (kappa * h - drift) * mask
    return g, h_qp, mask


def _cbf_constraints(hparams, obs, kappa, r_safe):
    if hparams.robot_type == "single_integrator":
        return _single_integrator_cbf_constraints(hparams, obs, kappa, r_safe)
    return _unicycle_cbf_constraints(hparams, obs, kappa, r_safe)


def solve_slack_qp(params, hparams, obs, nominal_xy, kappa, r_safe):
    if hparams.robot_type == "single_integrator":
        q, p = _single_integrator_qp_objective(hparams, obs, nominal_xy)
    else: 
        q, p = _unicycle_qp_objective(hparams, obs, kappa, nominal_xy)
    g, h_qp, _mask = _cbf_constraints(hparams, obs, kappa, r_safe) 
    return _solve_slack_qp(params, hparams, obs, q, p, g, h_qp)


def _common_diagnostics(nominal_xy, kappa, r_safe, safe_xy, slack, mask):
    slack_mean, slack_max = _masked_raw_stats(jnp.maximum(slack, 0.0), mask)


def lightweight_diagnostics(params, hparams, obs, nominal_xy, beta, r_safe, safe_xy, slack):
    del params, hparams
    _rel, _human_vel, _obstacle_radius, mask = _extract_blocks(obs)
    slack_mean, slack_max = _masked_raw_stats(jnp.maximum(slack, 0.0), mask)
    zero = jnp.zeros((obs.shape[0],), dtype=obs.dtype)
    correction = jnp.linalg.norm(safe_xy - nominal_xy, axis=1)
    return {
        "qp_penalty": slack_mean.astype(jnp.float32),
        "qp_penalty_max": slack_max.astype(jnp.float32),
        "qp_slack": slack_mean.astype(jnp.float32),
        "qp_slack_max": slack_max.astype(jnp.float32),
        "qp_correction": correction.astype(jnp.float32),
        "nominal_cbf_residual": zero.astype(jnp.float32),
        "nominal_cbf_residual_max": zero.astype(jnp.float32),
        "safe_cbf_residual": zero.astype(jnp.float32),
        "safe_cbf_residual_max": zero.astype(jnp.float32),
        "beta": beta.astype(jnp.float32),
        "r_safe": r_safe.astype(jnp.float32),
        "nominal_action_norm": jnp.linalg.norm(nominal_xy, axis=1).astype(jnp.float32),
        "safe_action_norm": jnp.linalg.norm(safe_xy, axis=1).astype(jnp.float32),
    }


def full_diagnostics(params, hparams, obs, nominal_xy, beta, r_safe, safe_xy, slack):
    g, h_qp, mask = _cbf_constraints(params["actor"], hparams, obs, beta, r_safe)
    nominal_raw = jnp.sum(g * nominal_xy[:, None, :], axis=2) - h_qp
    safe_raw = jnp.sum(g * safe_xy[:, None, :], axis=2) - h_qp
    nominal_mean, nominal_max = _masked_positive_stats(nominal_raw, mask)
    safe_mean, safe_max = _masked_positive_stats(safe_raw, mask)
    slack_mean, slack_max = _masked_raw_stats(jnp.maximum(slack, 0.0), mask)
    correction = jnp.linalg.norm(safe_xy - nominal_xy, axis=1)
    return {
        "qp_penalty": slack_mean.astype(jnp.float32),
        "qp_penalty_max": slack_max.astype(jnp.float32),
        "qp_slack": slack_mean.astype(jnp.float32),
        "qp_slack_max": slack_max.astype(jnp.float32),
        "qp_correction": correction.astype(jnp.float32),
        "nominal_cbf_residual": nominal_mean.astype(jnp.float32),
        "nominal_cbf_residual_max": nominal_max.astype(jnp.float32),
        "safe_cbf_residual": safe_mean.astype(jnp.float32),
        "safe_cbf_residual_max": safe_max.astype(jnp.float32),
        "beta": beta.astype(jnp.float32),
        "r_safe": r_safe.astype(jnp.float32),
        "nominal_action_norm": jnp.linalg.norm(nominal_xy, axis=1).astype(jnp.float32),
        "safe_action_norm": jnp.linalg.norm(safe_xy, axis=1).astype(jnp.float32),
    }
