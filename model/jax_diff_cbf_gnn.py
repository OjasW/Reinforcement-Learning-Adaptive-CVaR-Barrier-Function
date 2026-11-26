"""JAX diff- cbfactor with a shared GNN encoder"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.nn as jnn
import jax.numpy as jnp

from env.jax_robot import normalize_robot_type
from model import jax_actor_critic as actor_critic
from model import jax_cbf_qp
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
class DiffCBFGNNHParams:
    act_dim: int
    robot_type: str
    actor_act: str
    critic_act: str
    margin_base: float
    kappa_min:float
    kappa_max: float
    kappa_init: float
    learn_kappa: bool
    lookahead_distance: float
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
    require_config_value(model_cfg, "type", "diff_cbf_gnn", "model")
    require_config_value(actor_cfg, "coordinate_frame", "robot_body", "model.actor")

    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    obs_dim = int(model_cfg.obs_dim)
    if history_len <= 0:
        raise ValueError("diff_cbf_gnn requires history_len > 0")
    if obs_frame_dim < 6 or (obs_frame_dim - 6) % 6 != 0:
        raise ValueError(
            "diff_cbf_gnn requires obs_frame_dim = 6 + num_humans * 6"
        )
    if obs_dim != history_len * obs_frame_dim:
        raise ValueError(
            "diff_cbf_gnn requires obs_dim = history_len * obs_frame_dim"
        )
    if int(act_dim) != 2:
        raise ValueError("diff_cbf_gnn requires act_dim=2")

    num_humans = (obs_frame_dim - 6) // 6
    qp_obs_top_k = min(
        int(cfg_get(actor_cfg, "qp_obs_top_k", 5)), num_humans
    )
    if qp_obs_top_k <= 0:
        raise ValueError("diff_cbf_gnn requires qp_obs_top_k > 0")

    kappa_min = float(actor_cfg.kappa_min)
    kappa_max = float(actor_cfg.kappa_max)
    kappa_init = float(cfg_get(actor_cfg, "kappa_init", 0.5 * (kappa_min + kappa_max)))
    
    if not (0.0 < kappa_min < kappa_max):
        raise ValueError("diff_cbf_gnn requires 0 < kappa_min < kappa_max")
    if not (kappa_min < kappa_init < kappa_max):
        raise ValueError(
            "diff_cbf_gnn requires kappa_min < kappa_init < kappa_max"
        )

    margin_base = float(actor_cfg.margin_base)
    if margin_base < 0.0:
        raise ValueError("model.actor.margin_base must be >= 0")
    
    gnn_hidden_dim = int(cfg_get(actor_cfg, "gnn_hidden_dim", 64))
    gnn_latent_dim = int(cfg_get(actor_cfg, "gnn_latent_dim", 16))
    gnn_layers = int(cfg_get(actor_cfg, "gnn_layers", 3))
    position_scale = float(cfg_get(actor_cfg, "position_scale", 20.0))
    velocity_scale = float(cfg_get(actor_cfg, "velocity_scale", 1.0))
    if gnn_hidden_dim <= 0 or gnn_latent_dim <= 0 or gnn_layers <= 0:
        raise ValueError(
            "diff_cbf_gnn encoder dimensions and layer count must be > 0"
        )
    if position_scale <= 0.0 or velocity_scale <= 0.0:
        raise ValueError(
            "diff_cbf_gnn position_scale and velocity_scale must be > 0"
        )

    return DiffCBFGNNHParams(
        act_dim=int(act_dim),
        robot_type=normalize_robot_type(
            cfg_get(model_cfg, "robot_type", "unicycle")
        ),
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        margin_base=margin_base,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        kappa_init=kappa_init,
        learn_kappa=bool(cfg_get(actor_cfg, "learn_kappa", True)),
        lookahead_distance=float(
            cfg_get(actor_cfg, "lookahead_distance", 0.2)
        ),
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

def _kappa_logit(hparams):
    """Logit xi with kappa_min + (kappa_max - kappa_min) * sigmoid(xi) = kappa_init """
    frac = (hparams.kappa_init - hparams.kappa_min) / (hparams.kappa_max - hparams.kappa_min)
    return math.log(frac/ (1.0-frac))


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
            "diff_cbf_gnn actor and critic hidden dimensions must be > 0"
        )

    gat_keys, key = jax.random.split(key) 
    keys = jax.random.split(key, 8) # fc1, nominal_head x2, human_fc1, human_kappa_head, critic x3
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
            keys[0],
            hparams.gnn_latent_dim,
            hidden_dim,
            sqrt2,
            use_init_weights,
        ),
        "nominal_head": init_head(
            keys[1 : 3],
            hidden_dim,
            control_hidden_dim,
            act_dim,
            use_init_weights,
        ),
    }
    if hparams.learn_kappa:
        actor["human_fc1"] = init_linear(
            keys[3],
            hparams.gnn_latent_dim + hparams.gnn_hidden_dim + HUMAN_EDGE_DIM, # Change 2
            scalar_hidden_dim,
            sqrt2,
            use_init_weights,
        )
        kappa_head = init_linear(
            keys[4],
            scalar_hidden_dim,
            1, 
            sqrt2 * 0.01,
            use_init_weights,
        )
        # bias, so kappa starts at kappa_init for every obstacle (small weights --> near constant start)
        kappa_head["bias"] = jnp.full_like(kappa_head["bias"], _kappa_logit(hparams), dtype=jnp.float32)
        actor["human_kappa_head"] = kappa_head
    
    critic = {
        "fc1": init_linear(
            keys[5],
            hparams.gnn_latent_dim,
            critic_hidden_dim,
            sqrt2,
            use_init_weights,
        ),
        "fc21": init_linear(
            keys[6],
            critic_hidden_dim,
            critic_hidden_dim2,
            sqrt2,
            use_init_weights,
        ),
        "fc31": init_linear(
            keys[7],
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
            f"diff_cbf_gnn expects rank-2 observations, got ndim={obs.ndim}"
        )
    if obs.shape[-1] != hparams.obs_dim:
        raise ValueError(
            f"diff_cbf_gnn expects obs_dim={hparams.obs_dim}, got {obs.shape[-1]}"
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


def _kappa_from_logits(logits, hparams):
    """kappa_i = kappa_min + (kappa_max - kappa_min) * sigmoid(logits_i)"""
    return hparams.kappa_min + (hparams.kappa_max - hparams.kappa_min) * jnn.sigmoid(logits)


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

    
    if qp_context is None:
        qp_context = build_qp_context(obs, hparams)
    qp_obs = qp_context["qp_obs"]
    qp_topk_ids = qp_context["qp_topk_ids"]

    # per-human kappa (B, N) -> gather the top-K rows that enter the QP
    if hparams.learn_kappa:
        human_x = activation(dense(actor_params["human_fc1"], human_in), hparams.actor_act)
        kappa_logits_all = dense(actor_params["human_kappa_head"], human_x)[..., 0]  # (B, N)
        kappa_topk = _kappa_from_logits(_gather_topk(kappa_logits_all, qp_topk_ids), hparams)
    else:
        kappa_topk = jnp.full(qp_topk_ids.shape, hparams.kappa_init, dtype=jnp.float32)

    # fixed safe distance R_i = r_robot + r_human + margin_base (no learned margin)
    r_safe_topk = _gather_topk(_fixed_safe_distances(current_obs, hparams), qp_topk_ids)
    nominal_xy = _body_to_world(nominal_xy, qp_obs[:, 4])
    safe_xy, slack = jax_cbf_qp.solve_slack_qp(
        params,
        hparams,
        qp_obs,
        nominal_xy,
        kappa_topk,
        r_safe_topk,
    )
    return (
        latent,
        qp_obs,
        nominal_xy,
        kappa_topk,
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
        kappa,
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
        diag_fn = jax_cbf_qp.full_diagnostics if hparams.qp_full_diagnostics else jax_cbf_qp.lightweight_diagnostics
        diagnostics = diag_fn(
            params,
            hparams,
            qp_obs,
            nominal_xy,
            kappa,
            r_safe,
            safe_xy,
            slack,
        )        
        outputs = outputs + (diagnostics,)
    return outputs
