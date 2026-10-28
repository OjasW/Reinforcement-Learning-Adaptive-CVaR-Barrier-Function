"""[IMG] JAX diff-CVaR actor with a CNN image encoder.

What sees what:
1. CNN trunk (same Nature-DQN layout as ppo_cnn) sees ONLY the stacked policy images -> image latent.
2. Nominal head: image latent -> nominal action (body frame). Pixels only.
3. Risk heads (beta, r_safe): one row per QP top-k human.
     risk_head_input = "image_state": [image latent, that human's body-frame state (rel pos, rel vel, radius)]
     risk_head_input = "image":       [image latent] only -> every human gets the same logits
                                      (uniform beta split, one shared r_safe), so the network stays pixels-only.
   Why per-human heads: the image latent cannot tell which human sits in which obs slot, so per-slot
   heads (as in diff_cvar_mlp) would be meaningless. Same reason the GNN got per-human heads.
4. CVaR-CBF slack QP: uses the vector obs of the top-k nearest humans (positions, velocities, radii),
   exactly as diff_cvar_mlp/gnn. In a real robot these come from perception/tracking. The QP needs
   them by construction; the CNN does not see them.
5. Critic: shared image latent (default) or the asymmetric state critic from ppo_cnn (critic.input).

Notes:
- All network arrays float32 (trainer enables x64; the QP solver runs in float64 internally).
- Risk heads run on the K=qp_obs_top_k selected humans only (not all N slots): only those reach the QP,
  so the outputs and gradients are the same as computing N rows and gathering K, but cheaper.
- CNN, conv init and matmul layers are reused from model/jax_ppo_cnn.py; QP helpers from
  model/jax_diff_cvar_mlp.py (build_qp_context, beta budget, body/world rotations).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.nn as jnn
import jax.numpy as jnp

from env.jax_robot import normalize_robot_type
from model import jax_actor_critic as actor_critic
from model import jax_cvar_qp
from model import jax_ppo_cnn as _cnn
from model.jax_diff_cvar_mlp import (
    _allocate_beta_budget,
    _body_to_world,
    _uniform_beta_budget,
    _world_to_body,
    build_qp_context,
)
from model.jax_ppo_base import activation, cfg_get, init_linear, require_config_value

F32 = jnp.float32
HUMAN_STATE_DIM = 5  # [dx, dy, dvx, dvy, radius] in the robot body frame, scaled (same as the GNN edge features)
RISK_HEAD_INPUTS = ("image_state", "image")


@dataclass(frozen=True)
class DiffCVaRCNNHParams:
    act_dim: int
    robot_type: str
    actor_act: str
    critic_act: str
    # vector obs (QP + optional state critic)
    history_len: int
    obs_frame_dim: int
    obs_dim: int
    num_humans: int
    position_scale: float
    velocity_scale: float
    # image input
    image_size: int
    image_history_len: int
    # CNN trunk
    cnn_channels: tuple
    cnn_kernels: tuple
    cnn_strides: tuple
    cnn_latent_dim: int
    critic_input: str  # "image" | "state"
    encoder_hidden_dim: int
    encoder_latent_dim: int
    encoder_layers: int
    # risk heads
    risk_head_input: str  # "image_state" | "image"
    # CVaR-CBF QP (same fields as diff_cvar_mlp/gnn)
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
    qp_obs_top_k: int

    # same derived shapes as ppo_cnn (the reused CNN helpers read these)
    in_channels = _cnn.PPOCNNHParams.in_channels
    conv_out_hw = _cnn.PPOCNNHParams.conv_out_hw
    flat_dim = _cnn.PPOCNNHParams.flat_dim
    encoder_input_dim = _cnn.PPOCNNHParams.encoder_input_dim

    @property
    def risk_in_dim(self):
        extra = HUMAN_STATE_DIM if self.risk_head_input == "image_state" else 0
        return int(self.cnn_latent_dim) + extra


def hparams_from_config(model_cfg, critic_cfg, act_dim=2):
    actor_cfg = model_cfg.actor
    require_config_value(model_cfg, "type", "diff_cvar_cnn", "model")
    if cfg_get(actor_cfg, "qp_solver", "jax_qpth") != "jax_qpth":
        raise ValueError("this branch supports only model.actor.qp_solver=jax_qpth")

    # vector obs: still needed by the QP (current frame) and the optional state critic
    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    obs_dim = int(model_cfg.obs_dim)
    if history_len <= 0:
        raise ValueError("diff_cvar_cnn requires history_len > 0")
    if obs_frame_dim < 6 or (obs_frame_dim - 6) % 6 != 0:
        raise ValueError("diff_cvar_cnn requires obs_frame_dim = 6 + num_humans * 6")
    if obs_dim != history_len * obs_frame_dim:
        raise ValueError("diff_cvar_cnn requires obs_dim = history_len * obs_frame_dim")
    if int(act_dim) != 2:
        raise ValueError("diff_cvar_cnn requires act_dim=2")
    num_humans = (obs_frame_dim - 6) // 6

    # images (same rules as ppo_cnn)
    render_cfg = cfg_get(model_cfg, "render", None)
    if render_cfg is None:
        raise ValueError("diff_cvar_cnn needs a model.render block (the trainer only renders images when it exists)")
    image_size = int(cfg_get(render_cfg, "image_size", 84))
    image_history_len = int(cfg_get(model_cfg, "image_history_len", 4))
    if image_history_len not in (3, 4):
        raise ValueError("diff_cvar_cnn uses the latest 3 or 4 images (model.image_history_len)")
    cnn_cfg = cfg_get(actor_cfg, "cnn", None)
    channels = tuple(int(c) for c in cfg_get(cnn_cfg, "channels", (32, 64, 64)))
    kernels = tuple(int(k) for k in cfg_get(cnn_cfg, "kernels", (8, 4, 3)))
    strides = tuple(int(s) for s in cfg_get(cnn_cfg, "strides", (4, 2, 1)))
    if not (len(channels) == len(kernels) == len(strides) >= 1):
        raise ValueError("model.actor.cnn channels, kernels and strides must have the same length")
    critic_input = str(cfg_get(critic_cfg, "input", "image"))
    if critic_input not in ("image", "state"):
        raise ValueError(f"model.critic.input must be 'image' or 'state', got {critic_input!r}")
    risk_head_input = str(cfg_get(actor_cfg, "risk_head_input", "image_state"))
    if risk_head_input not in RISK_HEAD_INPUTS:
        raise ValueError(f"model.actor.risk_head_input must be one of {RISK_HEAD_INPUTS}, got {risk_head_input!r}")

    # QP (same checks as diff_cvar_mlp/gnn)
    qp_obs_top_k = min(int(cfg_get(actor_cfg, "qp_obs_top_k", 5)), num_humans)
    if qp_obs_top_k <= 0:
        raise ValueError("diff_cvar_cnn requires qp_obs_top_k > 0")
    beta_min = float(cfg_get(actor_cfg, "beta_min", 0.05))
    beta_max = float(actor_cfg.beta)
    if not (0.0 < beta_min < beta_max < 1.0):
        raise ValueError("diff_cvar_cnn requires 0 < beta_min < beta < 1")
    if qp_obs_top_k * beta_min > beta_max:
        raise ValueError("diff_cvar_cnn requires qp_obs_top_k * beta_min <= beta")
    margin_base = float(actor_cfg.margin_base)
    margin_extra_max = float(actor_cfg.margin_extra_max)
    if margin_base < 0.0 or margin_extra_max < 0.0:
        raise ValueError("model.actor.margin_base and margin_extra_max must be >= 0")

    position_scale = float(cfg_get(actor_cfg, "position_scale", 20.0))
    velocity_scale = float(cfg_get(actor_cfg, "velocity_scale", 1.0))
    if position_scale <= 0.0 or velocity_scale <= 0.0:
        raise ValueError("diff_cvar_cnn position_scale and velocity_scale must be > 0")

    hp = DiffCVaRCNNHParams(
        act_dim=int(act_dim),
        robot_type=normalize_robot_type(cfg_get(model_cfg, "robot_type", "unicycle")),
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        history_len=history_len,
        obs_frame_dim=obs_frame_dim,
        obs_dim=obs_dim,
        num_humans=num_humans,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
        image_size=image_size,
        image_history_len=image_history_len,
        cnn_channels=channels,
        cnn_kernels=kernels,
        cnn_strides=strides,
        cnn_latent_dim=int(cfg_get(cnn_cfg, "latent_dim", 512)),
        critic_input=critic_input,
        encoder_hidden_dim=int(cfg_get(critic_cfg, "encoder_hidden_dim", 64)),
        encoder_latent_dim=int(cfg_get(critic_cfg, "encoder_latent_dim", 16)),
        encoder_layers=int(cfg_get(critic_cfg, "encoder_layers", 3)),
        risk_head_input=risk_head_input,
        margin_base=margin_base,
        margin_extra_max=margin_extra_max,
        alpha=float(actor_cfg.alpha),
        beta_min=beta_min,
        beta_max=beta_max,
        learn_beta=bool(cfg_get(actor_cfg, "learn_beta", True)),
        learn_r_safe=bool(cfg_get(actor_cfg, "learn_r_safe", True)),
        lookahead_distance=float(cfg_get(actor_cfg, "lookahead_distance", 0.2)),
        gmm_lateral_ratio=float(actor_cfg.gmm_lateral_ratio),
        qp_max_iter=int(cfg_get(actor_cfg, "qp_max_iter", 40)),
        qp_eps=float(cfg_get(actor_cfg, "qp_eps", 1e-12)),
        qp_not_improved_lim=int(cfg_get(actor_cfg, "qp_not_improved_lim", 3)),
        qp_verbose=int(cfg_get(actor_cfg, "qp_verbose", -1)),
        qp_check_q_spd=bool(cfg_get(actor_cfg, "qp_check_q_spd", False)),
        qp_slack_weight=float(cfg_get(actor_cfg, "qp_slack_weight", 10.0)),
        qp_residual_reward_weight=float(actor_cfg.qp_residual_reward_weight),
        qp_full_diagnostics=bool(cfg_get(actor_cfg, "qp_full_diagnostics", False)),
        qp_obs_top_k=qp_obs_top_k,
    )
    if hp.conv_out_hw < 1:
        raise ValueError(f"CNN shrinks a {image_size}px image below 1 pixel; use larger images or smaller kernels")
    return hp


#--------- Params ---------#

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
    control_hidden_dim = int(cfg_get(actor_cfg, "control_hidden_dim", hidden_dim))
    scalar_hidden_dim = int(cfg_get(actor_cfg, "scalar_hidden_dim", hidden_dim))
    critic_hidden_dim = int(cfg_get(critic_cfg, "hidden_dim", 256))
    critic_hidden_dim2 = int(cfg_get(critic_cfg, "hidden_dim2", 256))
    if min(hidden_dim, control_hidden_dim, scalar_hidden_dim, critic_hidden_dim, critic_hidden_dim2) <= 0:
        raise ValueError("diff_cvar_cnn actor and critic hidden dimensions must be > 0")

    n_conv = len(hparams.cnn_channels)
    # conv layers, cnn fc, fc1, nominal x2, human_fc1, beta, rsafe, critic x3, state encoder
    keys = iter(jax.random.split(key, n_conv + 1 + 3 + 3 + 3 + hparams.encoder_layers))
    sqrt2 = math.sqrt(2.0)

    conv, in_ch = [], hparams.in_channels
    for out_ch, k in zip(hparams.cnn_channels, hparams.cnn_kernels):
        conv.append(_cnn._init_conv(next(keys), in_ch, out_ch, k, sqrt2, use_init_weights))
        in_ch = out_ch

    # CNN lives under params["actor"]: only "actor" and "critic" are trained (jax_ppo_core._trainable_params)
    actor = {
        "logstd": jnp.full((act_dim,), math.log(float(actor_cfg.action_std_init)), dtype=F32),
        "cnn": {"conv": conv, "fc": init_linear(next(keys), hparams.flat_dim, hparams.cnn_latent_dim, sqrt2, use_init_weights)},
        "fc1": init_linear(next(keys), hparams.cnn_latent_dim, hidden_dim, sqrt2, use_init_weights),
        "nominal_head": [
            init_linear(next(keys), hidden_dim, control_hidden_dim, sqrt2, use_init_weights),
            init_linear(next(keys), control_hidden_dim, act_dim, sqrt2 * 0.01, use_init_weights),
        ],
    }
    k_human, k_beta, k_rsafe = next(keys), next(keys), next(keys)
    if hparams.learn_beta or hparams.learn_r_safe:
        actor["human_fc1"] = init_linear(k_human, hparams.risk_in_dim, scalar_hidden_dim, sqrt2, use_init_weights)
    if hparams.learn_beta:
        actor["human_beta_head"] = init_linear(k_beta, scalar_hidden_dim, 1, sqrt2 * 0.01, use_init_weights)
    if hparams.learn_r_safe:
        actor["human_rsafe_head"] = init_linear(k_rsafe, scalar_hidden_dim, 1, sqrt2 * 0.01, use_init_weights)
    actor.update({
        "gmm_weights": jnp.asarray(actor_cfg.gmm_weights, dtype=F32),
        "gmm_variances": jnp.square(jnp.asarray(actor_cfg.gmm_stds, dtype=F32)),
    })

    critic_in = hparams.cnn_latent_dim
    critic = {}
    critic_keys = [next(keys) for _ in range(3)]
    if hparams.critic_input == "state":
        dims = ([hparams.encoder_input_dim] + [hparams.encoder_hidden_dim] * (hparams.encoder_layers - 1)
                + [hparams.encoder_latent_dim])
        critic["state_encoder"] = [init_linear(next(keys), i, o, sqrt2, use_init_weights) for i, o in zip(dims[:-1], dims[1:])]
        critic_in = hparams.encoder_latent_dim
    critic.update({
        "fc1": init_linear(critic_keys[0], critic_in, critic_hidden_dim, sqrt2, use_init_weights),
        "fc21": init_linear(critic_keys[1], critic_hidden_dim, critic_hidden_dim2, sqrt2, use_init_weights),
        "fc31": init_linear(critic_keys[2], critic_hidden_dim2, 1, 1.0, use_init_weights),
    })
    return {
        "actor": actor,
        "critic": critic,
        "action_low": jnp.asarray(action_low, dtype=F32),
        "action_high": jnp.asarray(action_high, dtype=F32),
    }


#--------- Forward ---------#

def _topk_human_state(qp_obs, hparams):
    """Body-frame features of the K selected humans from qp_obs -> (B, K, 5), zero for inactive rows."""
    qp_obs = jnp.asarray(qp_obs, dtype=F32)
    b, k = qp_obs.shape[0], hparams.qp_obs_top_k
    theta = qp_obs[:, 4]
    robot_vel = qp_obs[:, 2:4]
    blocks = qp_obs[:, 6:].reshape((b, k, 6))
    mask = jnp.clip(blocks[:, :, 5:6], 0.0, 1.0)
    rel = _world_to_body(blocks[:, :, 0:2], theta) / hparams.position_scale                     # robot - human
    rel_vel = _world_to_body(blocks[:, :, 2:4] - robot_vel[:, None, :], theta) / hparams.velocity_scale
    radius = blocks[:, :, 4:5] / hparams.position_scale
    return jnp.concatenate([rel, rel_vel, radius], axis=-1) * mask, blocks


def _topk_safe_distances(qp_obs, blocks, margin_logits, hparams):
    """r_safe for the K selected humans: robot_r + human_r + margin_base (+ learned extra)."""
    dtype = qp_obs.dtype
    base = qp_obs[:, 5:6] + blocks[:, :, 4] + jnp.asarray(hparams.margin_base, dtype=dtype)
    if margin_logits is None:
        return base
    return base + jnp.asarray(hparams.margin_extra_max, dtype=dtype) * jnn.sigmoid(margin_logits.astype(dtype))


def _actor_outputs(params, hparams, obs, image_obs, qp_context=None):
    actor_params = params["actor"]
    latent = _cnn._image_latent(params, hparams, image_obs)                                    # (B, L) pixels only

    # nominal action: pixels only
    hidden = activation(_cnn._lin(actor_params["fc1"], latent), hparams.actor_act)
    nominal_body = _cnn._mlp(actor_params["nominal_head"], hidden, hparams.actor_act)

    if qp_context is None:
        qp_context = build_qp_context(obs, hparams)
    qp_obs = qp_context["qp_obs"]
    human_state, blocks = _topk_human_state(qp_obs, hparams)                                   # (B, K, 5), (B, K, 6)
    k = hparams.qp_obs_top_k

    beta_logits = margin_logits = None
    if hparams.learn_beta or hparams.learn_r_safe:
        ctx = jnp.broadcast_to(latent[:, None, :], (latent.shape[0], k, latent.shape[-1]))
        human_in = jnp.concatenate([ctx, human_state], axis=-1) if hparams.risk_head_input == "image_state" else ctx
        human_x = activation(_cnn._lin(actor_params["human_fc1"], human_in), hparams.actor_act)
        if hparams.learn_beta:
            beta_logits = _cnn._lin(actor_params["human_beta_head"], human_x)[..., 0]             # (B, K)
        if hparams.learn_r_safe:
            margin_logits = _cnn._lin(actor_params["human_rsafe_head"], human_x)[..., 0]          # (B, K)

    beta_topk = (_allocate_beta_budget(beta_logits, blocks[:, :, 5], hparams) if hparams.learn_beta
                 else _uniform_beta_budget(blocks[:, :, 5], hparams))
    r_safe_topk = _topk_safe_distances(qp_obs, blocks, margin_logits, hparams)
    nominal_xy = _body_to_world(nominal_body, qp_obs[:, 4])
    safe_xy, slack = jax_cvar_qp.solve_slack_qp(params, hparams, qp_obs, nominal_xy, beta_topk, r_safe_topk)
    return latent, qp_obs, nominal_xy, beta_topk, r_safe_topk, safe_xy, slack


def safe_action_mean(params, hparams, obs, qp_context=None, image_obs=None):
    return _actor_outputs(params, hparams, obs, image_obs, qp_context=qp_context)[-2]


policy_action_mean = safe_action_mean
policy_action_to_env_action = actor_critic.policy_action_to_env_action


def critic_value(params, hparams, obs, image_obs=None):
    latent = _cnn._image_latent(params, hparams, image_obs) if hparams.critic_input == "image" else None
    return _cnn._critic_value(params, hparams, obs, latent)


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
    image_obs=None,
):
    latent, qp_obs, nominal_xy, beta, r_safe, safe_xy, slack = _actor_outputs(
        params, hparams, obs, image_obs, qp_context=qp_context
    )
    action, logprob, entropy, sampled_policy_action = actor_critic.action_logprob_entropy(
        params, safe_xy, action=action, noise=noise, policy_action=policy_action
    )
    value = _cnn._critic_value(params, hparams, obs, latent)
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    if return_diagnostics:
        diag_fn = jax_cvar_qp.full_diagnostics if hparams.qp_full_diagnostics else jax_cvar_qp.lightweight_diagnostics
        outputs = outputs + (diag_fn(params, hparams, qp_obs, nominal_xy, beta, r_safe, safe_xy, slack),)
    return outputs
