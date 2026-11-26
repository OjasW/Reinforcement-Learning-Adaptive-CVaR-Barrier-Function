"""Pure PPO actor/critic with a CNN.

Note:
1. Actor uses pixels only. 'obs' is still passed but not used.
2. Critic may use image or obs - asymmetric critic may hasten the training or be used if image does not give good results.
3. All arrays are float32 (the trainer enables x64). 
4. Linear layers use a real matmul: the repo's `dense`builds a (rows, out, in) tensor, which is 3136x512 per row here and far too large.

5. Network:  Nature-DQN trunk as in CleanRL ppo_atari
defaults: 84x84 input, T=4 frames -> 12 channels
conv 8x8 s4, 32 -> 20x20x32 | conv 4x4 s2, 64 -> 9x9x64 | conv 3x3 s1, 64 -> 7x7x64 | flatten 3136
    -> linear 512 (latent)
        -> actor: fc1 512->256, control head 256->256->2 (pre-tanh mean), learned logstd
        -> critic: fc1 512->256, fc21 256->256, fc31 256->1

6. Action sampling: 
tanh squashing and bounds are the shared helpers in model/jax_actor_critic.py, exactly as in ppo_mlp.
Hence, env actions stay inside [action_low, action_high].

"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from env.jax_robot import normalize_robot_type
from model import jax_actor_critic as actor_critic
from model.jax_ppo_base import (
    activation,
    cfg_get,
    init_linear,
    require_config_value,
)
from env.jax_image_renderer import history_to_cnn_input
from model import jax_ppo_mlp as _mlp_model

F32 = jnp.float32
IMAGE_CHANNELS = 3 # robot, goal, human

@dataclass(frozen=True)
class PPOCNNHParams:
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


    @property
    def in_channels(self):
        return IMAGE_CHANNELS * int(self.image_history_len)

    @property
    def conv_out_hw(self):
        hw = int(self.image_size)
        for k, s in zip(self.cnn_kernels, self.cnn_strides):
            hw = (hw - int(k)) // int(s) + 1    # VALID padding
        return hw

    @property
    def flat_dim(self):
        return self.conv_out_hw * self.conv_out_hw * int(self.cnn_channels[-1])

    @property
    def encoder_input_dim(self):
        return int(self.history_len) * (5 + 6 * int(self.num_humans))


def hparams_from_config(model_cfg, critic_cfg, act_dim=2):
    actor_cfg = model_cfg.actor
    require_config_value(model_cfg, "type", "ppo_cnn", "model")

    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    obs_dim = int(model_cfg.obs_dim)
    if history_len <= 0:
        raise ValueError("ppo_cnn requires history_len > 0")
    if obs_frame_dim < 6 or (obs_frame_dim - 6) % 6 != 0:
        raise ValueError("ppo_cnn requires obs_frame_dim = 6 + num_humans * 6")
    if obs_dim != history_len * obs_frame_dim:
        raise ValueError("ppo_cnn requires obs_dim = history_len * obs_frame_dim")
    if int(act_dim) != 2:
        raise ValueError("ppo_cnn requires act_dim=2")

    render_cfg = cfg_get(model_cfg, "render", None)
    if render_cfg is None:
        raise ValueError("ppo_cnn needs a model.render block (the trainer only renders images when it exists)")
    image_size = int(cfg_get(render_cfg, "image_size", 84))
    image_history_len = int(cfg_get(model_cfg, "image_history_len", 4))
    if image_history_len not in (3, 4):
        raise ValueError("ppo_cnn uses the latest 3 or 4 images (model.image_history_len)")

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

    hp = PPOCNNHParams(
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
    )
    if hp.conv_out_hw < 1:
        raise ValueError(f"CNN shrinks a {image_size}px image below 1 pixel; use larger images or smaller kernels")
    return hp


#--------- Layers ---------#
def _init_conv(key, in_ch, out_ch, kernel, gain, use_init_weights):
    fan_in = kernel * kernel* in_ch
    lin = init_linear(key, fan_in, out_ch, gain, use_init_weights)
    w = lin["weight"].reshape((out_ch, kernel, kernel, in_ch)).transpose((1, 2, 3, 0))
    return {"weight": w.astype(F32), "bias": lin["bias"].astype(F32)}

def _conv(layer, x, stride):
    y = jax.lax.conv_general_dilated(x, layer["weight"], window_strides=(stride, stride), padding="VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + layer["bias"]

def _lin(layer, x):
    return x @ layer["weight"].T + layer["bias"]

def _mlp(layers, x, act):
    for layer in layers[:-1]:
        x = activation(_lin(layer, x), act)
    return _lin(layers[-1], x)

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
    critic_hidden_dim = int(cfg_get(critic_cfg, "hidden_dim", 256))
    critic_hidden_dim2 = int(cfg_get(critic_cfg, "hidden_dim2", 256))
    if min(hidden_dim, control_hidden_dim, critic_hidden_dim, critic_hidden_dim2) <= 0:
        raise ValueError("ppo_cnn actor and critic hidden dimensions must be > 0")

    n_conv = len(hparams.cnn_channels)
    keys = iter(jax.random.split(key, n_conv + 1 + 3 + 3 + hparams.encoder_layers))
    sqrt2 = math.sqrt(2.0)
    
    conv, in_ch = [], hparams.in_channels
    for out_ch, k in zip(hparams.cnn_channels, hparams.cnn_kernels):
        conv.append(_init_conv(next(keys), in_ch, out_ch, k, sqrt2, use_init_weights))
        in_ch = out_ch
    
     # CNN lives under params["actor"]: only "actor" and "critic" are trained (jax_ppo_core._trainable_params)
    
    actor = {
        "logstd": jnp.full((act_dim,), math.log(float(actor_cfg.action_std_init)), dtype=F32),
        "cnn": {"conv": conv, "fc": init_linear(next(keys), hparams.flat_dim, hparams.cnn_latent_dim, sqrt2, use_init_weights)},
        "fc1": init_linear(next(keys), hparams.cnn_latent_dim, hidden_dim, sqrt2, use_init_weights),
        "control_head": [
            init_linear(next(keys), hidden_dim, control_hidden_dim, sqrt2, use_init_weights),
            init_linear(next(keys), control_hidden_dim, act_dim, sqrt2 * 0.01, use_init_weights),
        ],
    }
    critic_in = hparams.cnn_latent_dim
    critic = {}
    if hparams.critic_input == "state":
        dims = [hparams.encoder_input_dim] + [hparams.encoder_hidden_dim] * (hparams.encoder_layers - 1) + [hparams.encoder_latent_dim]
        critic["state_encoder"] = [init_linear(next(keys), i, o, sqrt2, use_init_weights) for i, o in zip(dims[:-1], dims[1:])]
        critic_in = hparams.encoder_latent_dim
    critic.update({
        "fc1": init_linear(next(keys), critic_in, critic_hidden_dim, sqrt2, use_init_weights),
        "fc21": init_linear(next(keys), critic_hidden_dim, critic_hidden_dim2, sqrt2, use_init_weights),
        "fc31" : init_linear(next(keys), critic_hidden_dim2, 1, 1.0, use_init_weights),
    })

    return {
        "actor": actor,
        "critic": critic,
        "action_low": jnp.asarray(action_low, F32),
        "action_high": jnp.asarray(action_high, F32),
    }


#--------- Forward ---------#

def _check_images(image_obs, hparams):
    if image_obs is None:
        raise ValueError("ppo_cnn needs image_obs (set model.render so the trainer renders images)")
    expected = (hparams.image_history_len, hparams.image_size, hparams.image_size, IMAGE_CHANNELS)
    if image_obs.ndim != 5 or tuple(image_obs.shape[1:]) != expected:
        raise ValueError(f"ppo_cnn expects image_obs (B,{','.join(map(str, expected))}), got {image_obs.shape}")

def _image_latent(params, hparams, image_obs):
    """uint8 (B,T,H,W,3) -> (B, cnn_latent_dim). The actor's only input."""
    _check_images(image_obs, hparams)
    x = history_to_cnn_input(image_obs).astype(F32)
    for layer, stride in zip(params["actor"]["cnn"]["conv"], hparams.cnn_strides):
        x = activation(_conv(layer, x, stride), hparams.actor_act)
    x = x.reshape((x.shape[0], -1))
    return activation(_lin(params["actor"]["cnn"]["fc"], x), hparams.actor_act)

def _actor_mean_from_latent(params, hparams, latent):
    hidden = activation(_lin(params["actor"]["fc1"], latent), hparams.actor_act)
    return _mlp(params["actor"]["control_head"], hidden, hparams.actor_act)


def _critic_value(params, hparams, obs, image_latent):
    if hparams.critic_input == "state":
        feats = _mlp_model._preprocess_obs(obs, hparams)
        latent = _mlp(params["critic"]["state_encoder"], feats, hparams.critic_act)
    else:
        latent = image_latent
    hidden = activation(_lin(params["critic"]["fc1"], latent), hparams.critic_act)
    hidden = activation(_lin(params["critic"]["fc21"], hidden), hparams.critic_act)
    return _lin(params["critic"]["fc31"], hidden).squeeze(-1)


def policy_action_mean(params, hparams, obs, image_obs=None):
    """Return the deterministic pre-tanh action mean."""
    del obs
    return _actor_mean_from_latent(params, hparams, _image_latent(params, hparams, image_obs))

policy_action_to_env_action = actor_critic.policy_action_to_env_action


def critic_value(params, hparams, obs, image_obs=None):
    latent = _image_latent(params, hparams, image_obs) if hparams.critic_input == "image" else None
    return _critic_value(params, hparams, obs, latent)


def get_action_and_value(
    params,
    hparams,
    obs,
    action=None,
    noise=None,
    policy_action=None,
    return_policy_action=False,
    image_obs=None,
):
    latent = _image_latent(params, hparams, image_obs)
    mean = _actor_mean_from_latent(params, hparams, latent)
    action, logprob, entropy, sampled_policy_action = actor_critic.action_logprob_entropy(
        params,
        mean,
        action=action,
        noise=noise,
        policy_action=policy_action,
    )
    value = _critic_value(params, hparams, obs, latent)
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    return outputs