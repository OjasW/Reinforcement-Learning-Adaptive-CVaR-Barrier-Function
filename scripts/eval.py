from pathlib import Path
import argparse
import glob
import json
import sys

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from eval_video import (  # noqa: E402
    _frame_to_rgb_array,
    _scalar,
    init_render_worker,
    render_jax_social_nav_frame,
    render_job,
    save_video,
)


class Evaluator:
    def __init__(
        self,
        save_dir,
        visualize=False,
        device=None,
        seeds=None,
        episodes=None,
        workers=None,  # [SPEED] processes drawing video frames in parallel (0 = draw in this process)
        vis_batch=16,  # [SPEED] episodes simulated together per compiled visualize rollout
    ):
        self.save_dir = Path(save_dir)
        self.visualize = bool(visualize)
        self.workers = workers
        self.vis_batch = max(1, int(vis_batch))
        self.seeds = list(seeds) if seeds is not None else None
        self.episodes = int(episodes) if episodes is not None else None
        self.video_save_dir = None
        self.config = OmegaConf.load(self.save_dir / "config.yaml")
        if device:
            self.config.device = str(device)
        self.jax_params = None
        self._jax_configured = False

        # [SPEED] built once per Evaluator and reused for every checkpoint: the same objects must be passed to
        # trainer.jax_ppo_eval each time so its compiled rollout is reused instead of recompiled.
        self._cache = {}

    def _configure_jax(self):
        if self._jax_configured:
            return
        from trainer.jax_platform import configure_jax_platform_for_device

        configure_jax_platform_for_device(getattr(self.config, "device", "cpu"))
        self._jax_configured = True

    def _model_module(self):
        from model import get_jax_model_module

        return get_jax_model_module(self.config.model)

    def _hparams(self):
        if "hparams" not in self._cache: # [SPEED] one object for all checkpoints (compile-cache key)
            self._configure_jax()
            model_module = self._model_module()
            self._cache["hparams"] = model_module.hparams_from_config(
                self.config.model,
                self.config.model.critic,
                act_dim=int(self.config.env.act_dim),
            )
        return self._cache["hparams"]

    def _env_cfg(self):
        if "env_cfg" not in self._cache:  # [SPEED] one object for all checkpoints (compile-cache key)
            self._configure_jax()
            from env.jax_social_nav import config_from_omegaconf

            self._cache["env_cfg"] = config_from_omegaconf(self.config)
        return self._cache["env_cfg"]

    def evaluate_one_ckpt(self, ckpt_path):
        self._configure_jax()
        from trainer.jax_checkpoint import load_jax_checkpoint

        ckpt_path = Path(ckpt_path)
        payload = load_jax_checkpoint(ckpt_path)
        self.jax_params = payload["params"]
        if self.visualize:
            self.video_save_dir = self.save_dir / f"visualize_{ckpt_path.stem}"
            self.video_save_dir.mkdir(parents=True, exist_ok=True)
        metrics = self.evaluate()
        return metrics

    def eval_all_ckpts(self, checkpoint=None):
        ckpt_paths = [Path(checkpoint)] if checkpoint else self._checkpoint_paths()
        if not ckpt_paths:
            raise FileNotFoundError(f"no ckpt_*.pkl files found under {self.save_dir}")

        results = {}
        for ckpt_path in ckpt_paths:
            print(f"Evaluating {ckpt_path}...")
            metrics = self.evaluate_one_ckpt(ckpt_path)
            results[ckpt_path.name] = metrics
            print(
                f"\treturn: {metrics['mean_return']:.2f} +/- {metrics['std_return']:.2f}, "
                f"success: {metrics['success_rate']:.2f}, collision: {metrics['collision_rate']:.2f}, "
                f"timeout: {metrics['timeout_rate']:.2f}"
            )

        results["aggregate"] = self._aggregate(results)
        if self.visualize:
            print(
                "visualize mode: skipped eval_results.json; "
                f"artifacts are under {self.video_save_dir}"
            )
            return results
        result_path = self.save_dir / "eval_results.json"
        result_path.write_text(json.dumps(results, indent=2))
        print(f"results: {result_path}")
        return results

    def evaluate(self):
        if self.jax_params is None:
            raise RuntimeError("evaluate_one_ckpt must load a JAX checkpoint before evaluate")
        if self.visualize:
            return self.evaluate_visualize()
        return self.evaluate_jax()

    def evaluate_jax(self):
        self._configure_jax()
        from trainer.jax_ppo_eval import evaluate_jax_env

        model_module = self._model_module()
        episodes = int(self.episodes) if self.episodes is not None else int(self.config.trainer.eval_episodes)
        return evaluate_jax_env(
            self.jax_params,
            self._hparams(),
            self._env_cfg(),  # [SPEED] cached object (was self.config, rebuilt on every call -> recompiled)
            episodes=episodes,
            policy_action_mean_fn=model_module.policy_action_mean,
            policy_action_to_env_action_fn=model_module.policy_action_to_env_action,
            seeds=self.seeds,
            include_paper_metrics=True,
        )

# --------- Visualize --------- #
# [SPEED] The old loop ran step_env, the policy and the obs update un-jitted, one Python call per step.
# step_env contains inner lax.scan / vmap calls, so every eager call recompiled them: ~2 s per env step
# Now each episode is one lax.scan over max_steps, many episodes run side by side (vmap), and it is compiled once per Evaluator.
# Randomness is unchanged: rng = PRNGKey(seed + ep), one split for the reset, one split per step, and each
# episode still steps a batch-1 env, so videos show the same scenarios as before.

    def _visualize_rollout_fn(self):
        if "vis_fn" in self._cache:
            return self._cache["vis_fn"]
        import jax
        import jax.numpy as jnp

        from env.jax_social_nav import fixed_order_policy_obs_from_state, reset_env, step_env

        env_cfg = self._env_cfg()
        model_module = self._model_module()
        hparams = self._hparams()
        history_len = int(self.config.model.history_len)
        has_qp_viz = hasattr(model_module, "build_qp_context") and hasattr(model_module, "safe_action_mean")
        max_steps = int(env_cfg.max_steps)

        def policy_step(params, policy_obs):
            if not has_qp_viz:
                policy_action = model_module.policy_action_mean(params, hparams, policy_obs)
                return model_module.policy_action_to_env_action(params, policy_action), {}
            qp_context = model_module.build_qp_context(policy_obs, hparams)
            policy_action = model_module.safe_action_mean(params, hparams, policy_obs, qp_context=qp_context)
            action, _logprob, _entropy, _value, diagnostics = model_module.get_action_and_value(
                params, hparams, policy_obs, policy_action=policy_action,
                return_diagnostics=True, qp_context=qp_context,
            )
            viz = {"qp_topk_ids": qp_context["qp_topk_ids"]}
            for name in ("r_safe", "beta", "alpha", "qp_slack", "qp_slack_max", "qp_correction"):
                if name in diagnostics:
                    viz[name] = diagnostics[name]
            return action, viz

        def episode(params, rng):
            rng, reset_key = jax.random.split(rng)
            state = reset_env(env_cfg, reset_key, 1)
            frame = fixed_order_policy_obs_from_state(env_cfg, state)
            history = jnp.repeat(frame[:, None, :], history_len, axis=1)

            def body(carry, _unused):
                state, history, rng = carry
                action, viz = policy_step(params, history.reshape((1, -1)))
                rng, step_key = jax.random.split(rng)
                next_state, reward, done, metrics = step_env(env_cfg, state, action, step_key)
                next_frame = fixed_order_policy_obs_from_state(env_cfg, next_state)
                next_history = jnp.concatenate([history[:, 1:, :], next_frame[:, None, :]], axis=1)
                rec = {
                    "state": state,  # state BEFORE this step (what the old trace stored)
                    "reward": reward[0],
                    "done": done[0] > 0.5,
                    "success": metrics["success"][0] > 0.5,
                    "collision": metrics["collision"][0] > 0.5,
                    "viz": viz,
                }
                return (next_state, next_history, rng), rec

            (final_state, _h, _rng), recs = jax.lax.scan(body, (state, history, rng), None, length=max_steps)
            return {"recs": recs, "final_state": final_state}

        fn = jax.jit(jax.vmap(episode, in_axes=(None, 0)))
        self._cache["vis_fn"] = fn
        return fn

    def _render_pool(self):
        """[SPEED] Worker processes for matplotlib frames (~90 ms each). 'spawn': never fork a JAX process."""
        import multiprocessing as mp
        import os

        n = self.workers if self.workers is not None else min(8, os.cpu_count() or 1)
        if n <= 0:
            return None
        env_cfg = self._env_cfg()
        ctx = mp.get_context("spawn")
        return ctx.Pool(n, initializer=init_render_worker, initargs=(self._env_lite(env_cfg),))

    @staticmethod
    def _env_lite(env_cfg):
        """Only what render_jax_social_nav_frame reads, as plain numpy/python (cheap to send to workers)."""
        from types import SimpleNamespace

        return SimpleNamespace(
            workspace_low=np.asarray(env_cfg.workspace_low, dtype=float),
            workspace_high=np.asarray(env_cfg.workspace_high, dtype=float),
            boundary_buffer=float(getattr(env_cfg, "boundary_buffer", 0.0)),
            robot_radius=float(env_cfg.robot_radius),
            robot_type=str(getattr(env_cfg, "robot_type", "unicycle")),
            dt=float(env_cfg.dt),
        )

    def _qp_viz_from(self, viz, k):
        """Rebuild the old per-step qp_viz dict from the recorded arrays (same fields and conversions)."""
        if not viz:
            return None
        hparams = self._hparams()

        def first_array(name):
            if name not in viz:
                return np.asarray(())
            value = np.asarray(viz[name][k])
            if value.size == 0:
                return value
            if value.ndim == 0:
                return value.reshape(1)
            return value[0]

        def first_scalar(name):
            value = np.asarray(viz[name][k]).reshape(-1)
            return float(value[0]) if value.size else 0.0

        return {
            "topk_ids": np.asarray(viz["qp_topk_ids"][k][0], dtype=int),
            "r_safe": np.asarray(first_array("r_safe"), dtype=float),
            "beta": np.asarray(first_array("beta"), dtype=float),
            "alpha": np.asarray(first_array("alpha"), dtype=float),
            "slack_mean": first_scalar("qp_slack"),
            "slack_max": first_scalar("qp_slack_max"),
            "qp_correction": first_scalar("qp_correction"),
            "lookahead_distance": (
                float(hparams.lookahead_distance) if hparams.robot_type == "unicycle" else 0.0
            ),
            "beta_min": float(getattr(hparams, "beta_min", 0.05)),
            "beta_total": float(getattr(hparams, "beta_max", 0.5)),
        }

    def evaluate_visualize(self):
        self._configure_jax()
        import jax
        from types import SimpleNamespace

        env_cfg = self._env_cfg()
        max_steps = int(env_cfg.max_steps)
        seeds = self.seeds if self.seeds is not None else list(range(100, 1000 + 1, 100))
        episodes = int(self.episodes) if self.episodes is not None else int(self.config.trainer.eval_episodes)
        rollout_seeds = [seed + ep for seed in seeds for ep in range(episodes)]
        fps = int(round(1.0 / float(env_cfg.dt)))
        env_lite = self._env_lite(env_cfg)
        rollout = self._visualize_rollout_fn()
        batch = min(self.vis_batch, len(rollout_seeds))

        returns = []
        success_count = collision_count = timeout_count = total_episodes = 0
        pool = self._render_pool()
        progress = tqdm(total=len(rollout_seeds), desc="Visualizing")
        try:
            for start in range(0, len(rollout_seeds), batch):
                chunk = rollout_seeds[start:start + batch]
                padded = chunk + [chunk[-1]] * (batch - len(chunk))  # same batch size -> no recompile
                keys = jax.numpy.stack([jax.random.PRNGKey(int(s)) for s in padded])
                out = jax.device_get(rollout(self.jax_params, keys))  # one device->host copy per chunk
                recs = out["recs"]

                for e, rollout_seed in enumerate(chunk):
                    done = np.asarray(recs["done"][e])
                    d = int(np.argmax(done)) if done.any() else None
                    n_items = max_steps if d is None else d + 1  # pre-step states shown (as before)

                    def state_at(k, e=e):
                        src = recs["state"] if k < max_steps else out["final_state"]
                        idx = (e, k) if k < max_steps else (e,)
                        pick = lambda a: np.asarray(a[idx])
                        return SimpleNamespace(
                            robot_state=pick(src.robot_state), human_positions=pick(src.human_positions),
                            human_radii=pick(src.human_radii), human_mask=pick(src.human_mask),
                            goal_pos=pick(src.goal_pos), step_count=pick(src.step_count),
                        )

                    states = [state_at(k) for k in range(n_items + (0 if d is None else 1))]
                    robot_xy = np.stack([np.asarray(st.robot_state[0][:2], dtype=float) for st in states])
                    rewards = np.asarray(recs["reward"][e])
                    viz_e = {name: np.asarray(arr[e]) for name, arr in recs["viz"].items()}  # this episode's QP diagnostics

                    jobs, total = [], 0.0
                    status, success, collision, timeout = "running", False, False, False
                    for k in range(n_items):
                        jobs.append((None, states[k], total, "running", robot_xy[: k + 1].copy(),
                                     self._qp_viz_from(viz_e, k)))
                        total += float(rewards[k])
                    if d is not None:
                        collision = bool(recs["collision"][e][d])
                        success = (not collision) and bool(recs["success"][e][d])
                        timeout = not (collision or success)
                        status = "collision" if collision else "success" if success else "timeout"
                        jobs.append((None, states[d + 1], total, status, robot_xy[: d + 2].copy(),
                                     self._qp_viz_from(viz_e, d)))

                    if pool is None:  # draw here
                        frames = [render_job((env_lite,) + job[1:]) for job in jobs]
                    else:  # [SPEED] draw in parallel, keep order
                        frames = list(pool.imap(render_job, jobs, chunksize=8))
                    video_path = self.video_save_dir / f"seed_{rollout_seed}_{status}.mp4"
                    save_video(video_path, frames, fps=fps)

                    returns.append(total)
                    total_episodes += 1
                    success_count += int(success)
                    collision_count += int(collision)
                    timeout_count += int(timeout or status == "running")
                    progress.update(1)
        finally:
            progress.close()
            if pool is not None:
                pool.close()
                pool.join()

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "success_rate": success_count / total_episodes,
            "collision_rate": collision_count / total_episodes,
            "timeout_rate": timeout_count / total_episodes,
        }

    def _checkpoint_paths(self):
        paths = [Path(path) for path in glob.glob(str(self.save_dir / "ckpt_*.pkl"))]
        paths.sort(key=lambda path: int(path.stem.split("_")[-1]))
        return paths

    @staticmethod
    def _aggregate(results):
        metrics = [value for key, value in results.items() if key != "aggregate"]
        aggregate = {
            "num_checkpoints": len(metrics),
            "min_return": float(min(metrics, key=lambda item: item["mean_return"])["mean_return"]),
            "max_return": float(max(metrics, key=lambda item: item["mean_return"])["mean_return"]),
            "mean_return": float(np.mean([m["mean_return"] for m in metrics])),
            "std_return": float(np.std([m["mean_return"] for m in metrics])),
            "mean_success_rate": float(np.mean([m["success_rate"] for m in metrics])),
            "mean_collision_rate": float(np.mean([m["collision_rate"] for m in metrics])),
            "mean_timeout_rate": float(np.mean([m["timeout_rate"] for m in metrics])),
            "max_success_rate": float(max(metrics, key=lambda item: item["success_rate"])["success_rate"]),
            "max_collision_rate": float(max(metrics, key=lambda item: item["collision_rate"])["collision_rate"]),
            "max_timeout_rate": float(max(metrics, key=lambda item: item["timeout_rate"])["timeout_rate"]),
            "min_success_rate": float(min(metrics, key=lambda item: item["success_rate"])["success_rate"]),
            "min_collision_rate": float(min(metrics, key=lambda item: item["collision_rate"])["collision_rate"]),
            "min_timeout_rate": float(min(metrics, key=lambda item: item["timeout_rate"])["timeout_rate"]),
        }
        for metric_name in ("traj_time", "min_dist"):
            values = [float(item[metric_name]) for item in metrics if item.get(metric_name) is not None]
            if values:
                aggregate[f"mean_{metric_name}"] = float(np.mean(values))
                aggregate[f"std_{metric_name}"] = float(np.std(values))
                aggregate[f"min_{metric_name}"] = float(np.min(values))
                aggregate[f"max_{metric_name}"] = float(np.max(values))
        return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True, help="Run directory containing config.yaml and ckpt_*.pkl")
    parser.add_argument("--checkpoint", default="", help="Evaluate one checkpoint instead of all ckpt_*.pkl files")
    parser.add_argument("--visualize", action="store_true", help="Save MP4 rollout videos with the JAX env renderer")
    parser.add_argument("--device", default="", choices=["", "cpu", "cuda", "auto"], help="Override device from config for eval/visualize")
    parser.add_argument("--seeds", default="", help="Comma-separated eval seeds; default is 100,200,...,1000")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per seed; default: 50")
    parser.add_argument("--workers", type=int, default=None, help="[SPEED] processes drawing video frames in parallel (default: CPU count, max 8; 0 = off)")
    parser.add_argument("--vis-batch", type=int, default=16, help="[SPEED] episodes simulated together in one compiled --visualize rollout")
    args = parser.parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()] if args.seeds else None
    Evaluator(
        args.save_dir,
        visualize=args.visualize,
        device=args.device or None,
        seeds=seeds,
        episodes=args.episodes,
        workers=args.workers,
        vis_batch=args.vis_batch,
    ).eval_all_ckpts(checkpoint=args.checkpoint or None)