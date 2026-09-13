"""Fixed-ID JAX rollout collection for supported PPO policy modules."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from env.jax_social_nav import fixed_order_policy_obs_from_state, reset_env, step_env
from trainer.jax_ppo_core import flat_time_batch


def _topk_mask(values, qp_obs):
    human_blocks = qp_obs[..., 6:].reshape(values.shape + (6,))
    return jnp.clip(human_blocks[..., 5], 0.0, 1.0).astype(values.dtype)


def _masked_topk_sum_count(values, qp_obs):
    mask = _topk_mask(values, qp_obs)
    return jnp.sum(values * mask), jnp.sum(mask, dtype=jnp.float32)


def _masked_topk_stats(values, qp_obs):
    mask = _topk_mask(values, qp_obs)
    active = mask > 0.0
    count = jnp.sum(mask, dtype=jnp.float32)
    total = jnp.sum(values * mask)
    denominator = jnp.maximum(count, 1.0)
    mean = total / denominator
    variance = jnp.sum(jnp.square(values - mean) * mask) / denominator
    minimum = jnp.where(
        count > 0.0, jnp.min(jnp.where(active, values, jnp.inf)), 0.0
    )
    maximum = jnp.where(
        count > 0.0, jnp.max(jnp.where(active, values, -jnp.inf)), 0.0
    )
    return minimum, maximum, jnp.sqrt(jnp.maximum(variance, 0.0))


def _beta_budget_error_max(beta, qp_obs, beta_budget):
    mask = _topk_mask(beta, qp_obs)
    beta_totals = jnp.sum(beta * mask, axis=-1)
    target = jnp.where(jnp.sum(mask, axis=-1) > 0.0, beta_budget, 0.0)
    return jnp.max(jnp.abs(beta_totals - target))


def _rollout_summary(transitions, use_qp, beta_budget=None):
    completed = jnp.isfinite(transitions["completed_return"])
    completed_returns = jnp.where(completed, transitions["completed_return"], 0.0)
    completed_ppo_returns = jnp.where(completed, transitions["completed_ppo_return"], 0.0)
    completed_lengths = jnp.where(completed, transitions["completed_length"], 0.0)
    summary = {
        "train_completed_count": completed.astype(jnp.int32).sum(),
        "train_return_sum": completed_returns.sum(),
        "train_ppo_return_sum": completed_ppo_returns.sum(),
        "train_length_sum": completed_lengths.sum(),
        "train_success_count": transitions["success"].astype(jnp.int32).sum(),
        "train_collision_count": transitions["collision"].astype(jnp.int32).sum(),
        "train_timeout_count": transitions["timeout"].astype(jnp.int32).sum(),
        "jax_env_min_clearance": jnp.nanmin(transitions["min_clearance"]),
        "boundary_violation_mean": jnp.mean(transitions["boundary_violation"]),
        "boundary_violation_max": jnp.max(transitions["boundary_violation"]),
        "robot_omega_delta_mean": jnp.mean(transitions["robot_omega_delta"]),
        "robot_omega_delta_max": jnp.max(transitions["robot_omega_delta"]),
        "robot_omega_delta_penalty_mean": jnp.mean(transitions["robot_omega_delta_penalty"]),
    }
    if use_qp:
        r_safe_sum, r_safe_count = _masked_topk_sum_count(
            transitions["r_safe"], transitions["qp_obs"]
        )
        summary.update({
            "qp_slack_sum": transitions["qp_slack"].sum(),
            "qp_slack_count": jnp.asarray(transitions["qp_slack"].size, dtype=jnp.float32),
            "qp_slack_max": jnp.max(transitions["qp_slack_max"]),
            "qp_correction_sum": transitions["qp_correction"].sum(),
            "qp_correction_count": jnp.asarray(transitions["qp_correction"].size, dtype=jnp.float32),
            "r_safe_sum": r_safe_sum,
            "r_safe_count": r_safe_count,
        })
        if "beta" in transitions:
            beta_active_min, beta_active_max, beta_active_std = (
                _masked_topk_stats(
                    transitions["beta"], transitions["qp_obs"]
                )
            )
            summary.update({
                "beta_active_min": beta_active_min,
                "beta_active_max": beta_active_max,
                "beta_active_std": beta_active_std,
                "beta_budget_error_max": _beta_budget_error_max(
                    transitions["beta"], transitions["qp_obs"], beta_budget
                ),
            })
        if "alpha" in transitions:
            alpha_active_min, alpha_active_max, alpha_active_std = (
                _masked_topk_stats(
                    transitions["alpha"], transitions["qp_obs"]
                )
            )
            summary.update({
                "alpha_active_min": alpha_active_min,
                "alpha_active_max": alpha_active_max,
                "alpha_active_std": alpha_active_std,
            })
        if "nominal_cbf_residual" in transitions:
            summary.update({
                "nominal_cbf_residual_sum": transitions["nominal_cbf_residual"].sum(),
                "nominal_cbf_residual_count": jnp.asarray(
                    transitions["nominal_cbf_residual"].size, dtype=jnp.float32
                ),
                "nominal_cbf_residual_max": jnp.max(transitions["nominal_cbf_residual_max"]),
                "safe_cbf_residual_sum": transitions["safe_cbf_residual"].sum(),
                "safe_cbf_residual_count": jnp.asarray(
                    transitions["safe_cbf_residual"].size, dtype=jnp.float32
                ),
                "safe_cbf_residual_max": jnp.max(transitions["safe_cbf_residual_max"]),
            })
    return summary


def _select_done(done_bool, reset_value, next_value):
    if next_value.ndim == 0:
        return next_value
    shape = (done_bool.shape[0],) + (1,) * (next_value.ndim - 1)
    return jnp.where(done_bool.reshape(shape), reset_value, next_value)


def _maybe_reset_done_envs(env_cfg, next_state, reset_key, done_bool, batch_size):
    def reset_done(_unused):
        reset_state = reset_env(env_cfg, reset_key, batch_size)
        return jax.tree_util.tree_map(
            lambda reset_value, next_value: _select_done(done_bool, reset_value, next_value),
            reset_state,
            next_state,
        )

    def keep_next(_unused):
        return next_state

    return jax.lax.cond(jnp.any(done_bool), reset_done, keep_next, operand=None)


def init_obs_history(env_cfg, env_state, history_len):
    frame = fixed_order_policy_obs_from_state(env_cfg, env_state)
    return jnp.repeat(frame[:, None, :], int(history_len), axis=1)


def _update_obs_history(history, next_frame, reset_mask):
    shifted = jnp.concatenate([history[:, 1:, :], next_frame[:, None, :]], axis=1)
    reset_history = jnp.repeat(next_frame[:, None, :], history.shape[1], axis=1)
    return jnp.where(reset_mask[:, None, None], reset_history, shifted)


def _update_ppo_episode_returns(running_returns, reward, done_bool):
    completed_returns = running_returns + reward
    next_returns = jnp.where(done_bool, 0.0, completed_returns)
    logged_returns = jnp.where(done_bool, completed_returns, jnp.nan)
    return next_returns.astype(jnp.float32), logged_returns.astype(jnp.float32)


def make_rollout_collector(model_module, use_qp):
    """Create a model-specific collector; ``use_qp`` is static before JIT tracing."""
    use_qp = bool(use_qp)

    def collect(
        params,
        hparams,
        env_cfg,
        env_state,
        rng,
        steps_per_env,
        global_step=0,
    ):
        nav_state, obs_history, ppo_episode_returns = env_state
        batch_size = int(nav_state.robot_state.shape[0])
        act_dim = int(hparams.act_dim)

        def body(carry, _unused):
            state, history, running_ppo_returns, key = carry
            key, noise_key, env_key, reset_key = jax.random.split(key, 4)
            policy_obs = history.reshape((batch_size, -1))
            noise = jax.random.normal(noise_key, (batch_size, act_dim), dtype=jnp.float32)
            if use_qp:
                qp_context = model_module.build_qp_context(policy_obs, hparams)
                env_action, logp, _entropy, value, policy_action, qp_diagnostics = (
                    model_module.get_action_and_value(
                        params,
                        hparams,
                        policy_obs,
                        noise=noise,
                        return_policy_action=True,
                        return_diagnostics=True,
                        qp_context=qp_context,
                    )
                )
            else:
                env_action, logp, _entropy, value, policy_action = model_module.get_action_and_value(
                    params,
                    hparams,
                    policy_obs,
                    noise=noise,
                    return_policy_action=True,
                )

            next_state, reward, done, metrics = step_env(env_cfg, state, env_action, env_key)
            if use_qp:
                reward = reward - float(hparams.qp_residual_reward_weight) * qp_diagnostics["qp_penalty"]
            done_bool = done > 0.5
            next_ppo_returns, completed_ppo_return = _update_ppo_episode_returns(
                running_ppo_returns,
                reward,
                done_bool,
            )
            next_state = _maybe_reset_done_envs(env_cfg, next_state, reset_key, done_bool, batch_size)
            next_frame = fixed_order_policy_obs_from_state(env_cfg, next_state)
            next_history = _update_obs_history(history, next_frame, done_bool)

            if use_qp:
                transition = {
                    "obs": policy_obs,
                    "qp_obs": qp_context["qp_obs"],
                    "qp_topk_ids": qp_context["qp_topk_ids"],
                    "act": env_action,
                    "policy_act": policy_action,
                    "logp": logp,
                    "rew": reward,
                    "done": done,
                    "val": value,
                    "completed_return": jnp.where(done_bool, metrics["episode_return"], jnp.nan),
                    "completed_ppo_return": completed_ppo_return,
                    "completed_length": jnp.where(done_bool, metrics["episode_length"], jnp.nan),
                    "success": metrics["success"],
                    "collision": metrics["collision"],
                    "timeout": metrics["timeout"],
                    "min_clearance": metrics["min_clearance"],
                    "boundary_violation": metrics["boundary_violation"],
                    "robot_omega_delta": metrics["robot_omega_delta"],
                    "robot_omega_delta_penalty": metrics["robot_omega_delta_penalty"],
                    "qp_slack": qp_diagnostics["qp_slack"],
                    "qp_slack_max": qp_diagnostics["qp_slack_max"],
                    "qp_correction": qp_diagnostics["qp_correction"],
                    "r_safe": qp_diagnostics["r_safe"],
                }
                if "beta" in qp_diagnostics:
                    transition["beta"] = qp_diagnostics["beta"]
                if "alpha" in qp_diagnostics:
                    transition["alpha"] = qp_diagnostics["alpha"]
                if hparams.qp_full_diagnostics:
                    transition.update({
                        "nominal_cbf_residual": qp_diagnostics["nominal_cbf_residual"],
                        "nominal_cbf_residual_max": qp_diagnostics["nominal_cbf_residual_max"],
                        "safe_cbf_residual": qp_diagnostics["safe_cbf_residual"],
                        "safe_cbf_residual_max": qp_diagnostics["safe_cbf_residual_max"],
                    })
            else:
                transition = {
                    "obs": policy_obs,
                    "act": env_action,
                    "policy_act": policy_action,
                    "logp": logp,
                    "rew": reward,
                    "done": done,
                    "val": value,
                    "completed_return": jnp.where(done_bool, metrics["episode_return"], jnp.nan),
                    "completed_ppo_return": completed_ppo_return,
                    "completed_length": jnp.where(done_bool, metrics["episode_length"], jnp.nan),
                    "success": metrics["success"],
                    "collision": metrics["collision"],
                    "timeout": metrics["timeout"],
                    "min_clearance": metrics["min_clearance"],
                    "boundary_violation": metrics["boundary_violation"],
                    "robot_omega_delta": metrics["robot_omega_delta"],
                    "robot_omega_delta_penalty": metrics["robot_omega_delta_penalty"],
                }
            return (next_state, next_history, next_ppo_returns, key), transition

        (next_state, next_history, next_ppo_returns, next_rng), transitions = jax.lax.scan(
            body,
            (nav_state, obs_history, ppo_episode_returns, rng),
            None,
            length=int(steps_per_env),
        )
        next_policy_obs = next_history.reshape((batch_size, -1))
        next_value = model_module.critic_value(params, hparams, next_policy_obs)
        beta_budget = (
            float(hparams.beta_max)
            if use_qp and hasattr(hparams, "beta_max")
            else None
        )
        summary = _rollout_summary(transitions, use_qp, beta_budget)
        if use_qp:
            batch = {
                "obs": flat_time_batch(transitions["obs"]),
                "qp_obs": flat_time_batch(transitions["qp_obs"]),
                "qp_topk_ids": flat_time_batch(transitions["qp_topk_ids"]),
                "act": flat_time_batch(transitions["act"]),
                "policy_act": flat_time_batch(transitions["policy_act"]),
                "logp": flat_time_batch(transitions["logp"]),
                "rew": transitions["rew"],
                "done": transitions["done"],
                "val": transitions["val"],
                "next_value": next_value,
                "global_step": jnp.asarray(global_step, dtype=jnp.int32) + int(steps_per_env) * batch_size,
                **summary,
            }
        else:
            batch = {
                "obs": flat_time_batch(transitions["obs"]),
                "act": flat_time_batch(transitions["act"]),
                "policy_act": flat_time_batch(transitions["policy_act"]),
                "logp": flat_time_batch(transitions["logp"]),
                "rew": transitions["rew"],
                "done": transitions["done"],
                "val": transitions["val"],
                "next_value": next_value,
                "global_step": jnp.asarray(global_step, dtype=jnp.int32) + int(steps_per_env) * batch_size,
                **summary,
            }
        return batch, (next_state, next_history, next_ppo_returns), next_rng

    return collect
