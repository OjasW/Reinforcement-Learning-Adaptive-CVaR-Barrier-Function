"""Differentiable CBF-QP actor with a CNN image encoder (Task 003 step 2: ground-truth geometry).

Note:
1. CNN trunk: same Nature-DQN layout as ppo_cnn. It only sees the stacked policy images -> image latent.
2. Kappa head: image latent -> K logits, one per QP row. Row j is always the j-th nearest human, because
   build_qp_context sorts the top-k by distance, so the image alone can tell the network which human each
   output is for. Per-human state is not concatenated here.
4. CBF slack QP: gets the ground-truth positions, velocities and radii of the top-k humans from the vector obs.
   That geometry enters only the QP constraints.
5. Critic: Two options - shared image latent (default) or the asymmetric state critic from ppo_cnn (critic.input).

- All network arrays are float32 (the trainer enables x64; the QP solver runs in float64 internally).
- CNN helpers come from model/jax_ppo_cnn.py, the QP context and kappa helpers from model/jax_diff_cbf_gnn.py,
  and the QP itself from model/jax_cbf_qp.py, so only the encoder and the kappa head differ from diff_cbf_gnn.

"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from env.jax_robot import normalize_robot_type
from model import jax_actor_critic as actor_critic
from model import jax_cbf_qp
from model import jax_ppo_cnn as _cnn
from model.jax_diff_cbf_gnn import (
    _body_to_world,
    _current_obs,
    _fixed_safe_distances,
    _gather_topk,
    _kappa_from_logits,
    _kappa_logit,
    build_qp_context,
)
from model.jax_ppo_base import activation, cfg_get, init_linear, require_config_value

F32 = jnp.float32


@dataclass(frozen=True)
class DiffCBFCNNHParams:
    act_dim: int
    robot_type: str
    actor_act: str
    critic_act: str
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
    critic_input: str # "image" / "state"
    # state-critic encoder (ppo_mlp sizes), used only when critic_input == "state"
    encoder_hidden_dim: int
    encoder_latent_dim: int
    encoder_layers: int
    # CBF QP
    margin_base: float
    kappa_min: float
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
    qp_obs_top_k: int
    in_channels = _cnn.PPOCNNHParams.in_channels
    conv_out_hw = _cnn.PPOCNNHParams.conv_out_hw
    flat_dim = _cnn.PPOCNNHParams.flat_dim
    encoder_input_dim = _cnn.PPOCNNHParams.encoder_input_dim


def hparams_from_config(model_cfg, critic_cfg, act_dim=2):
    actor_cfg = model_cfg.actor
    require_config_value(model_cfg, "type", "diff_cbf_cnn", "model")
    if cfg_get(actor_cfg, "qp_solver", "jax_qpth") != "jax_qpth":
        raise ValueError("this branch supports only model.actor.qp_solver=jax_qpth")


    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    obs_dim = int(model_cfg.obs_dim)
    if history_len <= 0:
        raise ValueError("diff_cbf_cnn requires history_len > 0")
    if obs_frame_dim < 6 or (obs_frame_dim - 6) % 6 != 0:
        raise ValueError("diff_cbf_cnn requires obs_frame_dim = 6 + num_humans * 6")
    if obs_dim != history_len * obs_frame_dim:
        raise ValueError("diff_cbf_cnn requires obs_dim = history_len * obs_frame_dim")
    if int(act_dim) != 2:
        raise ValueError("diff_cbf_cnn requires act_dim=2")
    num_humans = (obs_frame_dim - 6) // 6

    render_cfg = cfg_get(model_cfg, "render", None)
    if render_cfg is None:
        raise ValueError("diff_cbf_cnn needs a model.render block (the trainer only renders images when it exists)")
    image_size = int(cfg_get(render_cfg, "image_size", 84))
    image_history_len = int(cfg_get(model_cfg, "image_history_len", 4))
    if image_history_len not in (3, 4):
        raise ValueError("diff_cbf_cnn uses the latest 3 or 4 images (model.image_history_len)")

    cnn_cfg = cfg_get(actor_cfg, "cnn", None)
    channels = tuple(int(c) for c in cfg_get(cnn_cfg, "channels", (32, 64, 64)))
    kernels = tuple(int(k) for k in cfg_get(cnn_cfg, "kernels", (8, 4, 3)))
    strides = tuple(int(s) for s in cfg_get(cnn_cfg, "strides", (4, 2, 1)))
    latent_dim = int(cfg_get(cnn_cfg, "latent_dim", 512))
    if not (len(channels) == len(kernels) == len(strides) >= 1):
        raise ValueError("model.actor.cnn channels, kernels and strides must have the same length")

    critic_input = str(cfg_get(critic_cfg, "input", "image"))
    if critic_input not in ("image", "state"):
        raise ValueError(f"model.critic.input must be 'image' or 'state', got {critic_input!r}")

    qp_obs_top_k = min(int(cfg_get(actor_cfg, "qp_obs_top_k", 5)), num_humans)
    if qp_obs_top_k <= 0:
        raise ValueError("diff_cbf_cnn requires qp_obs_top_k > 0")
    
    kappa_min = float(actor_cfg.kappa_min)
    kappa_max = float(actor_cfg.kappa_max)
    kappa_init = float(cfg_get(actor_cfg, "kappa_init", 0.5 * (kappa_min + kappa_max)))
    if not (0.0 < kappa_min < kappa_init < kappa_max):
        raise ValueError("diff_cbf_cnn requires 0 < kappa_min < kappa_init < kappa_max")
    margin_base = float(actor_cfg.margin_base)
    if margin_base < 0.0:
        raise ValueError("model.actor.margin_base must be >= 0")



    hp = DiffCBFCNNHParams(
        act_dim=int(act_dim),
        robot_type=normalize_robot_type(cfg_get(model_cfg, "robot_type", "unicycle")),
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        history_len=history_len,
        obs_frame_dim=obs_frame_dim,
        obs_dim=obs_dim,
        num_humans=(obs_frame_dim - 6) // 6,
        position_scale=float(cfg_get(critic_cfg, "position_scale", 20.0)),
        velocity_scale=float(cfg_get(critic_cfg, "velocity_scale", 1.0)),
        image_size=image_size,
        image_history_len=image_history_len,
        cnn_channels=channels,
        cnn_kernels=kernels,
        cnn_strides=strides,
        cnn_latent_dim=latent_dim,
        critic_input=critic_input,
        encoder_hidden_dim=int(cfg_get(critic_cfg, "encoder_hidden_dim", 64)),
        encoder_latent_dim=int(cfg_get(critic_cfg, "encoder_latent_dim", 16)),
        encoder_layers=int(cfg_get(critic_cfg, "encoder_layers", 3)),
        margin_base=margin_base,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        kappa_init=kappa_init,
        learn_kappa=bool(cfg_get(actor_cfg, "learn_kappa", True)),
        lookahead_distance=float(cfg_get(actor_cfg, "lookahead_distance", 0.2)),
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
    if min(hidden_dim, control_hidden_dim, critic_hidden_dim, critic_hidden_dim2, scalar_hidden_dim) <= 0:
        raise ValueError("diff_cbf_cnn actor and critic hidden dimensions must be > 0")

    n_conv = len(hparams.cnn_channels)
    # conv layers, cnn fc, fc1, nominal x2, kappa x2, critic x3, state encoder
    keys = iter(jax.random.split(key, n_conv + 1 + 3 + 2 + 3 + hparams.encoder_layers))
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
    k_fc, k_head = next(keys), next(keys)
    if hparams.learn_kappa:
        actor["kappa_fc"] = init_linear(k_fc, hparams.cnn_latent_dim, scalar_hidden_dim, sqrt2, use_init_weights)
        kappa_head = init_linear(k_head, scalar_hidden_dim, hparams.qp_obs_top_k, sqrt2 * 0.01, use_init_weights)
        # bias, so every kappa starts at kappa_init (small weights -> near constant start)
        kappa_head["bias"] = jnp.full_like(kappa_head["bias"], _kappa_logit(hparams), dtype=F32)
        actor["kappa_head"] = kappa_head


    critic_in = hparams.cnn_latent_dim
    critic = {}
    critic_keys = [next(keys) for _ in range(3)]
    if hparams.critic_input == "state":
        dims = [hparams.encoder_input_dim] + [hparams.encoder_hidden_dim] * (hparams.encoder_layers - 1) + [hparams.encoder_latent_dim]
        critic["state_encoder"] = [init_linear(next(keys), i, o, sqrt2, use_init_weights) for i, o in zip(dims[:-1], dims[1:])]
        critic_in = hparams.encoder_latent_dim
    critic.update({
        "fc1": init_linear(critic_keys[0], critic_in, critic_hidden_dim, sqrt2, use_init_weights),
        "fc21": init_linear(critic_keys[1], critic_hidden_dim, critic_hidden_dim2, sqrt2, use_init_weights),
        "fc31" : init_linear(critic_keys[2], critic_hidden_dim2, 1, 1.0, use_init_weights),
    })

    return {
        "actor": actor,
        "critic": critic,
        "action_low": jnp.asarray(action_low, F32),
        "action_high": jnp.asarray(action_high, F32),
    }


#--------- Forward ---------#

def _network_outputs(params, hparams, image_obs):
    """The learned part of the actor. Takes images only, so it cannot see ground-truth state."""
    actor_params = params["actor"]
    latent = _cnn._image_latent(params, hparams, image_obs)
    hidden = activation(_cnn._lin(actor_params["fc1"], latent), hparams.actor_act)
    nominal_body = _cnn._mlp(actor_params["nominal_head"], hidden, hparams.actor_act)
    if hparams.learn_kappa:
        kappa_x = activation(_cnn._lin(actor_params["kappa_fc"], latent), hparams.actor_act)
        kappa_topk = _kappa_from_logits(_cnn._lin(actor_params["kappa_head"], kappa_x), hparams) #(B,K), nearest first
    else:
        kappa_topk = jnp.full((image_obs.shape[0], hparams.qp_obs_top_k), hparams.kappa_init, dtype=F32)
    return latent, nominal_body, kappa_topk


def _actor_outputs(params, hparams, obs, image_obs, qp_context=None):
    latent, nominal_body, kappa_topk = _network_outputs(params, hparams, image_obs)

    # ground truth geometry: used by QP
    if qp_context is None:
        qp_context = build_qp_context(obs, hparams)
    qp_obs = qp_context["qp_obs"]
    _history, current_obs = _current_obs(obs, hparams)
    r_safe_topk = _gather_topk(_fixed_safe_distances(current_obs, hparams), qp_context["qp_topk_ids"])
    nominal_xy = _body_to_world(nominal_body, qp_obs[:, 4])
    safe_xy, slack = jax_cbf_qp.solve_slack_qp(params, hparams, qp_obs, nominal_xy, kappa_topk, r_safe_topk)
    return latent, qp_obs, nominal_xy, kappa_topk, r_safe_topk, safe_xy, slack

def safe_action_mean(params, hparams, obs, qp_context=None, image_obs=None):
    return _actor_outputs(params, hparams, obs, image_obs, qp_context)[-2]


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
    latent, qp_obs, nominal_xy, kappa, r_safe, safe_xy, slack = _actor_outputs(params, hparams, obs, image_obs, qp_context=qp_context)
    action, logprob, entropy, sampled_policy_action = actor_critic.action_logprob_entropy(
        params,
        safe_xy,
        action=action,
        noise=noise,
        policy_action=policy_action,
    )
    value = _cnn._critic_value(params, hparams, obs, latent)
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    if return_diagnostics:
        diag_fn = jax_cbf_qp.full_diagnostics if hparams.qp_full_diagnostics else jax_cbf_qp.lightweight_diagnostics
        outputs = outputs + (diag_fn(params, hparams, qp_obs, nominal_xy, kappa, r_safe, safe_xy, slack),)
    return outputs