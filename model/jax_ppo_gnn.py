"""Pure PPO actor/critic with a graph neural network."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from env.jax_robot import normalize_robot_type
from model import jax_actor_critic as actor_critic
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


@dataclass(frozen=True)
class PPOGNNHParams:
    act_dim: int
    robot_type: str
    actor_act: str
    critic_act: str
    history_len: int
    obs_frame_dim: int
    obs_dim: int
    num_humans: int
    gnn_hidden_dim: int
    gnn_latent_dim: int
    gnn_layers: int
    position_scale: float
    velocity_scale: float

    # @property
    # def encoder_input_dim(self):
    #     processed_frame_dim = 5 + 6 * int(self.num_humans)
    #     return int(self.history_len) * processed_frame_dim


def hparams_from_config(model_cfg, critic_cfg, act_dim=2):
    actor_cfg = model_cfg.actor
    require_config_value(model_cfg, "type", "ppo_gnn", "model")
    require_config_value(actor_cfg, "coordinate_frame", "robot_body", "model.actor")

    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    obs_dim = int(model_cfg.obs_dim)
    if history_len <= 0:
        raise ValueError("ppo_gnn requires history_len > 0")
    if obs_frame_dim < 6 or (obs_frame_dim - 6) % 6 != 0:
        raise ValueError("ppo_gnn requires obs_frame_dim = 6 + num_humans * 6")
    if obs_dim != history_len * obs_frame_dim:
        raise ValueError("ppo_gnn requires obs_dim = history_len * obs_frame_dim")
    if int(act_dim) != 2:
        raise ValueError("ppo_gnn requires act_dim=2")

    gnn_hidden_dim = int(cfg_get(actor_cfg, "gnn_hidden_dim", 64))
    gnn_latent_dim = int(cfg_get(actor_cfg, "gnn_latent_dim", 16))
    gnn_layers = int(cfg_get(actor_cfg, "gnn_layers", 3))
    position_scale = float(cfg_get(actor_cfg, "position_scale", 20.0))
    velocity_scale = float(cfg_get(actor_cfg, "velocity_scale", 1.0))
    if gnn_hidden_dim <= 0 or gnn_latent_dim <= 0 or gnn_layers <= 0:
        raise ValueError("ppo_gnn encoder dimensions and layer count must be > 0")
    if position_scale <= 0.0 or velocity_scale <= 0.0:
        raise ValueError("ppo_gnn position_scale and velocity_scale must be > 0")

    return PPOGNNHParams(
        act_dim=int(act_dim),
        robot_type=normalize_robot_type(cfg_get(model_cfg, "robot_type", "unicycle")),
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        history_len=history_len,
        obs_frame_dim=obs_frame_dim,
        obs_dim=obs_dim,
        num_humans=(obs_frame_dim - 6) // 6,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_latent_dim=gnn_latent_dim,
        gnn_layers=gnn_layers,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
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
    control_hidden_dim = int(cfg_get(actor_cfg, "control_hidden_dim", hidden_dim))
    critic_hidden_dim = int(cfg_get(critic_cfg, "hidden_dim", 256))
    critic_hidden_dim2 = int(cfg_get(critic_cfg, "hidden_dim2", 256))
    if min(hidden_dim, control_hidden_dim, critic_hidden_dim, critic_hidden_dim2) <= 0:
        raise ValueError("ppo_gnn actor and critic hidden dimensions must be > 0")

    gat_keys, key = jax.random.split(key)
    keys = jax.random.split(key, 6)         # fc1, control_head x2, critic x3
    gnn_end = 0
    sqrt2 = math.sqrt(2.0)
    actor = {
        "logstd": jnp.full((act_dim,), math.log(float(actor_cfg.action_std_init)), dtype=jnp.float32),
        "gat": _init_gat_params(gat_keys, 4, 7, hparams.gnn_hidden_dim, hparams.gnn_latent_dim, hparams.gnn_layers, use_init_weights),
        "fc1": init_linear(
            keys[gnn_end],
            hparams.gnn_latent_dim,
            hidden_dim,
            sqrt2,
            use_init_weights,
        ),
        "control_head": init_head(
            keys[gnn_end + 1 : gnn_end + 3],
            hidden_dim,
            control_hidden_dim,
            act_dim,
            use_init_weights,
        ),
    }
    critic = {
        "fc1": init_linear(
            keys[gnn_end + 3],
            hparams.gnn_latent_dim,
            critic_hidden_dim,
            sqrt2,
            use_init_weights,
        ),
        "fc21": init_linear(
            keys[gnn_end + 4],
            critic_hidden_dim,
            critic_hidden_dim2,
            sqrt2,
            use_init_weights,
        ),
        "fc31": init_linear(
            keys[gnn_end + 5],
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


def visualize_obs(obs, hparams, b=0, **kwargs):
    """Build the graph for a batch of observations and draw sample b."""
    from visualize_graph import visualize_graph

    obs = jnp.asarray(obs, dtype=jnp.float32)
    if obs.ndim == 1:
        obs = obs[None]                      # _graph_ expects (B, obs_dim)
    node_features, edge_attr, src, dst, _ = _graph_(obs, hparams)
    return visualize_graph(
        node_features, edge_attr, src, dst,
        history_len=hparams.history_len,
        num_humans=hparams.num_humans,
        b=b, **kwargs,
    )

def _shared_latent(params, hparams, obs):
    nf, ea, src, dst, robot_idx = _graph_(obs, hparams)
    return gat_encode(params["actor"]["gat"], nf, ea, src, dst, robot_idx, hparams.actor_act)


def _actor_mean_from_latent(params, hparams, latent):
    hidden = activation(dense(params["actor"]["fc1"], latent), hparams.actor_act)
    return mlp(params["actor"]["control_head"], hidden, hparams.actor_act)


def _critic_value_from_latent(params, hparams, latent):
    hidden = activation(dense(params["critic"]["fc1"], latent), hparams.critic_act)
    hidden = activation(dense(params["critic"]["fc21"], hidden), hparams.critic_act)
    return dense(params["critic"]["fc31"], hidden).squeeze(-1)


def policy_action_mean(params, hparams, obs):
    """Return the deterministic pre-tanh action mean."""
    latent = _shared_latent(params, hparams, obs)
    return _actor_mean_from_latent(params, hparams, latent)


policy_action_to_env_action = actor_critic.policy_action_to_env_action


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
):
    latent = _shared_latent(params, hparams, obs)
    mean = _actor_mean_from_latent(params, hparams, latent)
    action, logprob, entropy, sampled_policy_action = actor_critic.action_logprob_entropy(
        params,
        mean,
        action=action,
        noise=noise,
        policy_action=policy_action,
    )
    value = _critic_value_from_latent(params, hparams, latent)
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    return outputs
