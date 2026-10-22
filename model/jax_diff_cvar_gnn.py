"""JAX diff-CVaR actor with a shared GNN encoder"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.nn as jnn
import jax.numpy as jnp

from env.jax_robot import normalize_robot_type
from model import jax_actor_critic as actor_critic
from model import jax_cvar_qp
from model.jax_graph import _graph_
from model.jax_gat import _init_gat_params, gat_encode
from model.jax_ppo_base import (
    activation,
    cfg_get,
    dense,
    init_head,
    init_linear,
    mlp,
    require_config_value,
)

HUMAN_EDGE_DIM = 5 # [dx, dy, vx, vy, radius] in robot frame # Change 1

@dataclass(frozen=True)
class DiffCVaRGNNHParams:
    act_dim: int
    robot_type: str
    actor_act: str
    critic_act: str
    margin_base: float
    margin_extra_max: float
    alpha: float
    beta_min: float
    beta_max: float
    learn_beta: bool
    learn_r_safe: bool
    lookahead_distance: float
    gmm_lateral_ratio: float
    qp_max_iter: int
    qp_eps: float
    qp_not_improved_lim: int
    qp_verbose: int
    qp_check_q_spd: bool
    qp_slack_weight: float
    qp_residual_reward_weight: float
    qp_full_diagnostics: bool
    history_len: int
    obs_frame_dim: int
    obs_dim: int
    num_humans: int
    gnn_hidden_dim: int
    gnn_latent_dim: int
    gnn_layers: int
    position_scale: float
    velocity_scale: float
    qp_obs_top_k: int


def hparams_from_config(model_cfg, critic_cfg, act_dim=2):
    actor_cfg = model_cfg.actor
    if cfg_get(actor_cfg, "qp_solver", "jax_qpth") != "jax_qpth":
        raise ValueError("this branch supports only model.actor.qp_solver=jax_qpth")
    require_config_value(model_cfg, "type", "diff_cvar_gnn", "model")
    require_config_value(
        actor_cfg, "coordinate_frame", "robot_body", "model.actor"
    )

    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    obs_dim = int(model_cfg.obs_dim)
    if history_len <= 0:
        raise ValueError("diff_cvar_gnn requires history_len > 0")
    if obs_frame_dim < 6 or (obs_frame_dim - 6) % 6 != 0:
        raise ValueError(
            "diff_cvar_gnn requires obs_frame_dim = 6 + num_humans * 6"
        )
    if obs_dim != history_len * obs_frame_dim:
        raise ValueError(
            "diff_cvar_gnn requires obs_dim = history_len * obs_frame_dim"
        )
    if int(act_dim) != 2:
        raise ValueError("diff_cvar_gnn requires act_dim=2")

    num_humans = (obs_frame_dim - 6) // 6
    qp_obs_top_k = min(
        int(cfg_get(actor_cfg, "qp_obs_top_k", 5)), num_humans
    )
    if qp_obs_top_k <= 0:
        raise ValueError("diff_cvar_gnn requires qp_obs_top_k > 0")

    beta_min = float(cfg_get(actor_cfg, "beta_min", 0.05))
    beta_max = float(actor_cfg.beta)
    if not (0.0 < beta_min < beta_max < 1.0):
        raise ValueError("diff_cvar_gnn requires 0 < beta_min < beta < 1")
    if qp_obs_top_k * beta_min > beta_max:
        raise ValueError(
            "diff_cvar_gnn requires qp_obs_top_k * beta_min <= beta"
        )

    margin_base = float(actor_cfg.margin_base)
    margin_extra_max = float(actor_cfg.margin_extra_max)
    if margin_base < 0.0:
        raise ValueError("model.actor.margin_base must be >= 0")
    if margin_extra_max < 0.0:
        raise ValueError("model.actor.margin_extra_max must be >= 0")

    gnn_hidden_dim = int(cfg_get(actor_cfg, "gnn_hidden_dim", 64))
    gnn_latent_dim = int(cfg_get(actor_cfg, "gnn_latent_dim", 16))
    gnn_layers = int(cfg_get(actor_cfg, "gnn_layers", 3))
    position_scale = float(cfg_get(actor_cfg, "position_scale", 20.0))
    velocity_scale = float(cfg_get(actor_cfg, "velocity_scale", 1.0))
    if gnn_hidden_dim <= 0 or gnn_latent_dim <= 0 or gnn_layers <= 0:
        raise ValueError(
            "diff_cvar_gnn encoder dimensions and layer count must be > 0"
        )
    if position_scale <= 0.0 or velocity_scale <= 0.0:
        raise ValueError(
            "diff_cvar_gnn position_scale and velocity_scale must be > 0"
        )

    return DiffCVaRGNNHParams(
        act_dim=int(act_dim),
        robot_type=normalize_robot_type(
            cfg_get(model_cfg, "robot_type", "unicycle")
        ),
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        margin_base=margin_base,
        margin_extra_max=margin_extra_max,
        alpha=float(actor_cfg.alpha),
        beta_min=beta_min,
        beta_max=beta_max,
        learn_beta=bool(cfg_get(actor_cfg, "learn_beta", True)),
        learn_r_safe=bool(cfg_get(actor_cfg, "learn_r_safe", True)),
        lookahead_distance=float(
            cfg_get(actor_cfg, "lookahead_distance", 0.2)
        ),
        gmm_lateral_ratio=float(actor_cfg.gmm_lateral_ratio),
        qp_max_iter=int(cfg_get(actor_cfg, "qp_max_iter", 40)),
        qp_eps=float(cfg_get(actor_cfg, "qp_eps", 1e-12)),
        qp_not_improved_lim=int(
            cfg_get(actor_cfg, "qp_not_improved_lim", 3)
        ),
        qp_verbose=int(cfg_get(actor_cfg, "qp_verbose", -1)),
        qp_check_q_spd=bool(cfg_get(actor_cfg, "qp_check_q_spd", False)),
        qp_slack_weight=float(cfg_get(actor_cfg, "qp_slack_weight", 10.0)),
        qp_residual_reward_weight=float(actor_cfg.qp_residual_reward_weight),
        qp_full_diagnostics=bool(
            cfg_get(actor_cfg, "qp_full_diagnostics", False)
        ),
        history_len=history_len,
        obs_frame_dim=obs_frame_dim,
        obs_dim=obs_dim,
        num_humans=num_humans,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_latent_dim=gnn_latent_dim,
        gnn_layers=gnn_layers,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
        qp_obs_top_k=qp_obs_top_k,
    )


def init_params_from_config(
    key,
    model_cfg,
    critic_cfg,
    act_dim,
    action_low,
    action_high,
    use_init_weights=True,
):
    actor_cfg = model_cfg.actor
    hparams = hparams_from_config(model_cfg, critic_cfg, act_dim=act_dim)
    hidden_dim = int(cfg_get(actor_cfg, "hidden_dim", 256))
    control_hidden_dim = int(
        cfg_get(actor_cfg, "control_hidden_dim", hidden_dim)
    )
    scalar_hidden_dim = int(
        cfg_get(actor_cfg, "scalar_hidden_dim", hidden_dim)
    )
    critic_hidden_dim = int(cfg_get(critic_cfg, "hidden_dim", 256))
    critic_hidden_dim2 = int(cfg_get(critic_cfg, "hidden_dim2", 256))
    if min(
        hidden_dim,
        control_hidden_dim,
        scalar_hidden_dim,
        critic_hidden_dim,
        critic_hidden_dim2,
    ) <= 0:
        raise ValueError(
            "diff_cvar_gnn actor and critic hidden dimensions must be > 0"
        )

    gat_keys, key = jax.random.split(key) 
    keys = jax.random.split(key, 9) # fc1, nominal_head x2, human_fc1, human_beta_head, human_rsafe_head, critic x3
    gnn_end = 0
    sqrt2 = math.sqrt(2.0)
    actor = {
        "logstd": jnp.full(
            (act_dim,),
            math.log(float(actor_cfg.action_std_init)),
            dtype=jnp.float32,
        ),
        "gat": _init_gat_params(
            gat_keys, 4, 7, hparams.gnn_hidden_dim, hparams.gnn_latent_dim, hparams.gnn_layers, use_init_weights
        ),
        "fc1": init_linear(
            keys[gnn_end],
            hparams.gnn_latent_dim,
            hidden_dim,
            sqrt2,
            use_init_weights,
        ),
        "nominal_head": init_head(
            keys[gnn_end + 1 : gnn_end + 3],
            hidden_dim,
            control_hidden_dim,
            act_dim,
            use_init_weights,
        ),
    }
    if hparams.learn_beta or hparams.learn_r_safe:
        actor["human_fc1"] = init_linear(
            keys[gnn_end + 3],
            hparams.gnn_latent_dim + hparams.gnn_hidden_dim + HUMAN_EDGE_DIM, # Change 2
            scalar_hidden_dim,
            sqrt2,
            use_init_weights,
        )
    if hparams.learn_beta:
        actor["human_beta_head"] = init_linear(
            keys[gnn_end + 4],
            scalar_hidden_dim,
            1,     # hparams.num_humans, # Change 3
            sqrt2 * 0.01,
            use_init_weights,
        )
    if hparams.learn_r_safe:
        actor["human_rsafe_head"] = init_linear(
            keys[gnn_end + 5],
            scalar_hidden_dim,
            1,    # hparams.num_humans, # Change 4
            sqrt2 * 0.01,
            use_init_weights,
        )
    actor.update({
        "gmm_weights": jnp.asarray(
            actor_cfg.gmm_weights, dtype=jnp.float32
        ),
        "gmm_variances": jnp.square(
            jnp.asarray(actor_cfg.gmm_stds, dtype=jnp.float32)
        ),
    })

    critic = {
        "fc1": init_linear(
            keys[gnn_end + 6],
            hparams.gnn_latent_dim,
            critic_hidden_dim,
            sqrt2,
            use_init_weights,
        ),
        "fc21": init_linear(
            keys[gnn_end + 7],
            critic_hidden_dim,
            critic_hidden_dim2,
            sqrt2,
            use_init_weights,
        ),
        "fc31": init_linear(
            keys[gnn_end + 8],
            critic_hidden_dim2,
            1,
            1.0,
            use_init_weights,
        ),
    }
    return {
        "actor": actor,
        "critic": critic,
        "action_low": jnp.asarray(action_low, dtype=jnp.float32),
        "action_high": jnp.asarray(action_high, dtype=jnp.float32),
    }


def _body_to_world(vectors, theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    x = vectors[:, 0]
    y = vectors[:, 1]
    return jnp.stack([c * x - s * y, s * x + c * y], axis=1)


def _obs_history(obs, hparams):
    obs = jnp.asarray(obs, dtype=jnp.float32)
    if obs.ndim != 2:
        raise ValueError(
            f"diff_cvar_gnn expects rank-2 observations, got ndim={obs.ndim}"
        )
    if obs.shape[-1] != hparams.obs_dim:
        raise ValueError(
            f"diff_cvar_gnn expects obs_dim={hparams.obs_dim}, got {obs.shape[-1]}"
        )
    return obs.reshape(
        (obs.shape[0], hparams.history_len, hparams.obs_frame_dim)
    )


def _current_obs(obs, hparams):
    history = _obs_history(obs, hparams)
    return history, history[:, -1, :]


def _select_qp_topk(current_obs, human_ids, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:].reshape(
        (batch_size, hparams.num_humans, 6)
    )
    mask = jnp.clip(blocks[:, :, 5], 0.0, 1.0) > 0.5
    distance = jnp.linalg.norm(blocks[:, :, 0:2], axis=2)
    sort_key = jnp.where(mask, distance, jnp.inf)
    _values, topk_ids = jax.lax.top_k(
        -sort_key, hparams.qp_obs_top_k
    )
    batch_idx = jnp.arange(batch_size)[:, None]
    topk_blocks = blocks[batch_idx, topk_ids]
    selected_human_ids = human_ids[batch_idx, topk_ids]
    qp_obs = jnp.concatenate(
        [current_obs[:, :6], topk_blocks.reshape((batch_size, -1))],
        axis=1,
    )
    return qp_obs, selected_human_ids


def build_qp_context(obs, hparams):
    """Build observation-only QP context for reuse during PPO updates."""
    _history, current_obs = _current_obs(obs, hparams)
    batch_size = current_obs.shape[0]
    human_ids = jnp.broadcast_to(
        jnp.arange(hparams.num_humans, dtype=jnp.int32)[None, :],
        (batch_size, hparams.num_humans),
    )
    qp_obs, qp_topk_ids = _select_qp_topk(
        current_obs, human_ids, hparams
    )
    return {
        "qp_obs": qp_obs,
        "qp_topk_ids": qp_topk_ids.astype(jnp.int32),
    }


def _gather_topk(values, qp_topk_ids):
    batch_idx = jnp.arange(values.shape[0])[:, None]
    return values[batch_idx, qp_topk_ids.astype(jnp.int32)]


def _allocate_beta_budget(beta_logits, mask, hparams):
    active = jnp.clip(mask, 0.0, 1.0) > 0.5
    active_float = active.astype(beta_logits.dtype)
    masked_logits = jnp.where(
        active, beta_logits, jnp.asarray(-1e9, dtype=beta_logits.dtype)
    )
    weights = jnn.softmax(masked_logits, axis=-1) * active_float
    weights = weights / jnp.maximum(
        jnp.sum(weights, axis=-1, keepdims=True), 1e-8
    )
    valid_count = jnp.sum(active_float, axis=-1, keepdims=True)
    remaining = jnp.maximum(
        hparams.beta_max - hparams.beta_min * valid_count, 0.0
    )
    beta = hparams.beta_min + remaining * weights
    return jnp.where(
        active,
        beta,
        jnp.asarray(hparams.beta_min, dtype=beta_logits.dtype),
    )


def _uniform_beta_budget(mask, hparams):
    active = jnp.clip(mask, 0.0, 1.0) > 0.5
    dtype = mask.dtype
    active_count = jnp.sum(active.astype(dtype), axis=-1, keepdims=True)
    beta = jnp.asarray(hparams.beta_max, dtype=dtype) / jnp.maximum(
        active_count, 1.0
    )
    return jnp.where(
        active, beta, jnp.asarray(hparams.beta_min, dtype=dtype)
    )


def _safe_distances_from_margin_logits(current_obs, margin_logits, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:].reshape(
        (batch_size, hparams.num_humans, 6)
    )
    robot_radius = current_obs[:, 5:6]
    human_radii = blocks[:, :, 4]
    margin_base = jnp.asarray(
        hparams.margin_base, dtype=current_obs.dtype
    )
    margin_extra_max = jnp.asarray(
        hparams.margin_extra_max, dtype=current_obs.dtype
    )
    learned_margin = margin_base + margin_extra_max * jnn.sigmoid(
        margin_logits
    )
    return robot_radius + human_radii + learned_margin


def _fixed_safe_distances(current_obs, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:].reshape(
        (batch_size, hparams.num_humans, 6)
    )
    robot_radius = current_obs[:, 5:6]
    human_radii = blocks[:, :, 4]
    margin_base = jnp.asarray(
        hparams.margin_base, dtype=current_obs.dtype
    )
    return robot_radius + human_radii + margin_base


def _shared_latent(params, hparams, obs):
    nf, ea, src, dst, robot_idx = _graph_(obs, hparams)
    return gat_encode(params["actor"]["gat"], nf, ea, src, dst, robot_idx, hparams.actor_act)


def _encode_with_humans(params, hparams, obs):
    """Robot latent plus one feature row per human slot (latest timestep)."""
    nf, ea, src, dst, robot_idx = _graph_(obs, hparams)
    latent, h = gat_encode(
        params["actor"]["gat"], nf, ea, src, dst, robot_idx, hparams.actor_act, return_nodes=True
    )
    n_h, t_last = hparams.num_humans, hparams.history_len - 1
    human_h = h[:, robot_idx + 1 : robot_idx + 1 + n_h, :]                      # (B, N, H)
    sp0 = t_last * (n_h + 1)                                                    # spatial edges of the latest step
    human_e = ea[:, sp0 : sp0 + n_h, :HUMAN_EDGE_DIM]                           # (B, N, 5)
    ctx = jnp.broadcast_to(latent[:, None, :], (latent.shape[0], n_h, latent.shape[-1]))
    return latent, jnp.concatenate([ctx, human_h, human_e], axis=-1)            # (B, N, L+H+5)
 
# Change 5


def _actor_outputs(params, hparams, obs, qp_context=None):
    latent, human_in = _encode_with_humans(params, hparams, obs)
    _history, current_obs = _current_obs(obs, hparams)
    actor_params = params["actor"]
    hidden = activation(
        dense(actor_params["fc1"], latent), hparams.actor_act
    )
    nominal_xy = mlp(
        actor_params["nominal_head"], hidden, hparams.actor_act
    )

    if hparams.learn_beta or hparams.learn_r_safe:
        human_x = activation(
            dense(actor_params["human_fc1"], human_in), hparams.actor_act
        )
    if hparams.learn_beta:
        beta_logits_all = dense(actor_params["human_beta_head"], human_x)[..., 0]    # (B, N)
    if hparams.learn_r_safe:
        margin_logits_all = dense(actor_params["human_rsafe_head"], human_x)[..., 0] # (B, N)
        r_safe_all = _safe_distances_from_margin_logits(
            current_obs, margin_logits_all, hparams
        )
    else:
        r_safe_all = _fixed_safe_distances(current_obs, hparams)

    if qp_context is None:
        qp_context = build_qp_context(obs, hparams)
    qp_obs = qp_context["qp_obs"]
    qp_topk_ids = qp_context["qp_topk_ids"]
    qp_blocks = qp_obs[:, 6:].reshape(
        (qp_obs.shape[0], hparams.qp_obs_top_k, 6)
    )
    if hparams.learn_beta:
        beta_logits_topk = _gather_topk(
            beta_logits_all, qp_topk_ids
        )
        beta_topk = _allocate_beta_budget(
            beta_logits_topk, qp_blocks[:, :, 5], hparams
        )
    else:
        beta_topk = _uniform_beta_budget(
            qp_blocks[:, :, 5], hparams
        )
    r_safe_topk = _gather_topk(r_safe_all, qp_topk_ids)
    nominal_xy = _body_to_world(nominal_xy, qp_obs[:, 4])
    safe_xy, slack = jax_cvar_qp.solve_slack_qp(
        params,
        hparams,
        qp_obs,
        nominal_xy,
        beta_topk,
        r_safe_topk,
    )
    return (
        latent,
        qp_obs,
        nominal_xy,
        beta_topk,
        r_safe_topk,
        safe_xy,
        slack,
    )


def safe_action_mean(params, hparams, obs, qp_context=None):
    outputs = _actor_outputs(
        params, hparams, obs, qp_context=qp_context
    )
    return outputs[-2]


policy_action_mean = safe_action_mean
policy_action_to_env_action = actor_critic.policy_action_to_env_action


def _critic_value_from_latent(params, hparams, latent):
    hidden = activation(
        dense(params["critic"]["fc1"], latent), hparams.critic_act
    )
    hidden = activation(
        dense(params["critic"]["fc21"], hidden), hparams.critic_act
    )
    return dense(params["critic"]["fc31"], hidden).squeeze(-1)


def critic_value(params, hparams, obs):
    latent = _shared_latent(params, hparams, obs)
    return _critic_value_from_latent(params, hparams, latent)


def get_action_and_value(
    params,
    hparams,
    obs,
    action=None,
    noise=None,
    policy_action=None,
    return_policy_action=False,
    return_diagnostics=False,
    qp_context=None,
):
    (
        latent,
        qp_obs,
        nominal_xy,
        beta,
        r_safe,
        safe_xy,
        slack,
    ) = _actor_outputs(params, hparams, obs, qp_context=qp_context)
    action, logprob, entropy, sampled_policy_action = (
        actor_critic.action_logprob_entropy(
            params,
            safe_xy,
            action=action,
            noise=noise,
            policy_action=policy_action,
        )
    )
    value = _critic_value_from_latent(params, hparams, latent)
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    if return_diagnostics:
        if hparams.qp_full_diagnostics:
            diagnostics = jax_cvar_qp.full_diagnostics(
                params,
                hparams,
                qp_obs,
                nominal_xy,
                beta,
                r_safe,
                safe_xy,
                slack,
            )
        else:
            diagnostics = jax_cvar_qp.lightweight_diagnostics(
                params,
                hparams,
                qp_obs,
                nominal_xy,
                beta,
                r_safe,
                safe_xy,
                slack,
            )
        outputs = outputs + (diagnostics,)
    return outputs
