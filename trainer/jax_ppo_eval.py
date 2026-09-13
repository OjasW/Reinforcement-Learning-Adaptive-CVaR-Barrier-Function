"""Deterministic evaluation for the supported JAX GNN policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from env.jax_social_nav import (
    config_from_omegaconf,
    fixed_order_policy_obs_from_state,
    reset_env_with_keys,
    step_env,
)


DEFAULT_EVAL_SEEDS = tuple(range(100, 1001, 100))


def _eval_seed_values(episodes, seeds=None):
    base_seeds = DEFAULT_EVAL_SEEDS if seeds is None else tuple(int(seed) for seed in seeds)
    if not base_seeds:
        raise ValueError("at least one evaluation seed is required")
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    seed_values = []
    for seed in base_seeds:
        for episode in range(episodes):
            seed_values.append(int(seed) + episode)
    return np.asarray(seed_values, dtype=np.uint32)


def _select_tree_by_active(active, old_tree, new_tree):
    def select(old_value, new_value):
        shape = (active.shape[0],) + (1,) * (new_value.ndim - 1)
        return jnp.where(active.reshape(shape), new_value, old_value)

    return jax.tree_util.tree_map(select, old_tree, new_tree)


def _fixed_slot_policy_obs_from_state(env_cfg, state, model_humans):
    batch_size = state.robot_state.shape[0]
    model_humans = int(model_humans)
    if model_humans <= 0:
        raise ValueError("model_humans must be > 0")

    robot = jnp.concatenate(
        [
            state.robot_state[:, 0:2] - state.goal_pos,
            state.robot_vel,
            state.robot_state[:, 2:3],
            jnp.full((batch_size, 1), env_cfg.robot_radius, dtype=state.robot_state.dtype),
        ],
        axis=1,
    )
    rel = state.robot_state[:, None, 0:2] - state.human_positions
    blocks = jnp.concatenate(
        [
            rel,
            state.human_vels,
            state.human_radii[..., None],
            state.human_mask[..., None],
        ],
        axis=-1,
    )
    dists = jnp.linalg.norm(rel, axis=-1)
    sort_key = jnp.where(state.human_mask > 0.5, dists, jnp.inf)
    selected = jnp.argsort(sort_key, axis=1)
    selected_blocks = jnp.take_along_axis(blocks, selected[..., None], axis=1)
    valid = jnp.take_along_axis(state.human_mask, selected, axis=1) > 0.5
    selected_blocks = selected_blocks.at[:, :, 5].set(valid.astype(selected_blocks.dtype))
    selected_blocks = selected_blocks[:, :model_humans, :]
    pad_count = model_humans - int(selected_blocks.shape[1])
    if pad_count > 0:
        pad = jnp.zeros((batch_size, pad_count, 6), dtype=selected_blocks.dtype)
        selected_blocks = jnp.concatenate([selected_blocks, pad], axis=1)
    return jnp.concatenate([robot, selected_blocks.reshape((batch_size, -1))], axis=1).astype(jnp.float32)


def _policy_obs_frame_from_state(env_cfg, state, hparams):
    obs_frame_dim = int(getattr(hparams, "obs_frame_dim", 0))
    if obs_frame_dim >= 6 and (obs_frame_dim - 6) % 6 == 0:
        model_humans = (obs_frame_dim - 6) // 6
        if 0 < model_humans != int(env_cfg.num_humans):
            return _fixed_slot_policy_obs_from_state(env_cfg, state, model_humans)
    return fixed_order_policy_obs_from_state(env_cfg, state)


def _evaluate_seed_values(
    params,
    hparams,
    env_cfg,
    seed_values,
    policy_action_mean_fn=None,
    policy_action_to_env_action_fn=None,
):
    """Evaluate all requested episodes as one batched JAX env rollout."""
    if policy_action_mean_fn is None or policy_action_to_env_action_fn is None:
        raise ValueError("policy functions must be provided by the selected model")

    keys = jnp.stack([jax.random.PRNGKey(int(seed)) for seed in seed_values], axis=0)
    state = reset_env_with_keys(env_cfg, keys)
    batch_size = int(seed_values.shape[0])
    frame = _policy_obs_frame_from_state(env_cfg, state, hparams)
    history = jnp.repeat(frame[:, None, :], int(hparams.history_len), axis=1)

    init_done = jnp.zeros((batch_size,), dtype=bool)
    init_returns = jnp.zeros((batch_size,), dtype=jnp.float32)
    init_success = jnp.zeros((batch_size,), dtype=bool)
    init_collision = jnp.zeros((batch_size,), dtype=bool)
    init_timeout = jnp.zeros((batch_size,), dtype=bool)
    init_lengths = jnp.zeros((batch_size,), dtype=jnp.int32)
    init_min_clearance = jnp.full((batch_size,), jnp.inf, dtype=jnp.float32)
    init_key = jax.random.PRNGKey(0)
    max_steps = jnp.asarray(int(env_cfg.max_steps), dtype=jnp.int32)
    init_step = jnp.asarray(0, dtype=jnp.int32)

    def cond(carry):
        (
            _state,
            _history,
            done,
            _returns,
            _success,
            _collision,
            _timeout,
            _lengths,
            _min_clearance,
            _key,
            step,
        ) = carry
        return (step < max_steps) & (~jnp.all(done))

    def body(carry):
        (
            curr_state,
            curr_history,
            done,
            returns,
            success,
            collision,
            timeout,
            lengths,
            min_clearance,
            key,
            step,
        ) = carry
        key, step_key = jax.random.split(key)
        policy_obs = curr_history.reshape((batch_size, -1))
        policy_action = policy_action_mean_fn(params, hparams, policy_obs)
        env_action = policy_action_to_env_action_fn(params, policy_action)
        next_state, reward, step_done, metrics = step_env(env_cfg, curr_state, env_action, step_key)

        active = ~done
        terminal = active & (step_done > 0.5)
        raw_collision = metrics["collision"] > 0.5
        raw_success = metrics["success"] > 0.5
        terminal_collision = terminal & raw_collision
        terminal_success = terminal & (~raw_collision) & raw_success
        terminal_timeout = terminal & (~terminal_collision) & (~terminal_success)
        next_done = done | terminal
        returns = returns + jnp.where(active, reward, 0.0)
        success = success | terminal_success
        collision = collision | terminal_collision
        timeout = timeout | terminal_timeout
        lengths = lengths + active.astype(jnp.int32)
        min_clearance = jnp.minimum(
            min_clearance,
            jnp.where(active, metrics["min_clearance"], jnp.inf),
        )
        selected_state = _select_tree_by_active(active, curr_state, next_state)
        next_frame = _policy_obs_frame_from_state(env_cfg, selected_state, hparams)
        shifted_history = jnp.concatenate([curr_history[:, 1:, :], next_frame[:, None, :]], axis=1)
        next_history = jnp.where(active[:, None, None], shifted_history, curr_history)
        return (
            selected_state,
            next_history,
            next_done,
            returns,
            success,
            collision,
            timeout,
            lengths,
            min_clearance,
            key,
            step + 1,
        )

    (
        _state,
        _history,
        done,
        returns,
        success,
        collision,
        timeout,
        lengths,
        min_clearance,
        _key,
        _step,
    ) = jax.lax.while_loop(
        cond,
        body,
        (
            state,
            history,
            init_done,
            init_returns,
            init_success,
            init_collision,
            init_timeout,
            init_lengths,
            init_min_clearance,
            init_key,
            init_step,
        ),
    )
    timeout = timeout | (~done & ~success & ~collision)
    return returns, success, collision, timeout, lengths, min_clearance


def _safe_mean_std(values):
    finite = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.std(finite))


def _paper_metric_summary(success_np, lengths_np, min_clearance_np, dt):
    success_mask = success_np > 0.5
    traj_times = lengths_np.astype(np.float32) * float(dt)
    traj_time_mean, traj_time_std = _safe_mean_std(traj_times[success_mask])
    min_dist_mean, min_dist_std = _safe_mean_std(min_clearance_np[success_mask])
    return {
        "traj_time": traj_time_mean,
        "std_traj_time": traj_time_std,
        "min_dist": min_dist_mean,
        "std_min_dist": min_dist_std,
    }


def evaluate_jax_env(
    params,
    hparams,
    env_cfg_or_config,
    episodes=None,
    seeds=None,
    policy_action_mean_fn=None,
    safe_action_mean_fn=None,
    policy_action_to_env_action_fn=None,
    history_len=None,
    include_paper_metrics=False,
):
    """Run deterministic evaluation using fixed-ID observation history."""
    env_cfg = env_cfg_or_config
    if not hasattr(env_cfg_or_config, "max_steps"):
        env_cfg = config_from_omegaconf(env_cfg_or_config)
    if policy_action_mean_fn is None and safe_action_mean_fn is not None:
        policy_action_mean_fn = safe_action_mean_fn

    seed_values = _eval_seed_values(1 if episodes is None else episodes, seeds=seeds)
    returns, success, collision, timeout, lengths, min_clearance = _evaluate_seed_values(
        params,
        hparams,
        env_cfg,
        seed_values,
        policy_action_mean_fn=policy_action_mean_fn,
        policy_action_to_env_action_fn=policy_action_to_env_action_fn,
    )
    returns_np = np.asarray(returns, dtype=np.float32)
    success_np = np.asarray(success, dtype=np.float32)
    collision_np = np.asarray(collision, dtype=np.float32)
    timeout_np = np.asarray(timeout, dtype=np.float32)
    lengths_np = np.asarray(lengths, dtype=np.int32)
    min_clearance_np = np.asarray(min_clearance, dtype=np.float32)
    total = int(returns_np.size)
    metrics = {
        "mean_return": float(np.mean(returns_np)),
        "std_return": float(np.std(returns_np)),
        "success_rate": float(success_np.sum()) / total,
        "collision_rate": float(collision_np.sum()) / total,
        "timeout_rate": float(timeout_np.sum()) / total,
    }
    if include_paper_metrics:
        metrics.update(_paper_metric_summary(success_np, lengths_np, min_clearance_np, float(env_cfg.dt)))
    return metrics
