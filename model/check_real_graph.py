"""Draw the GNN graph for real environment observations.

Run from the repo root:
    python check_real_graph.py                       # 6 humans, 8 steps, sample 0
    python check_real_graph.py --humans 20 --steps 12 --sample 3 --masked

Checks done before drawing: edge_index matches the numpy template, and the
tensors are finite and float32.
"""

import argparse
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("add", lambda *v: sum(v), replace=True)
OmegaConf.register_new_resolver("mul", lambda *v: math.prod(v), replace=True)

from env.jax_social_nav import config_from_omegaconf, reset_env, step_env
from model import get_jax_model_module
from model.jax_graph import _graph_
from trainer.jax_ppo_rollout import _update_obs_history, init_obs_history
from env.jax_social_nav import fixed_order_policy_obs_from_state
from model.visualize_graph import build_edge_index, visualize_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=int, default=6)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--masked", action="store_true", help="also draw masked nodes/edges")
    ap.add_argument("--labels", action="store_true", help="label edges with |(dx,dy)|")
    ap.add_argument("--out", default="real_graph.png")
    ap.add_argument("--robot", default="unicycle")
    args = ap.parse_args()

    with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["model=ppo_gnn", f"robot={args.robot}", f"env.humans.num_humans={args.humans}"],
        )
    model = get_jax_model_module(cfg.model)
    hp = model.hparams_from_config(cfg.model, cfg.model.critic, act_dim=2)

    env_cfg = config_from_omegaconf(cfg)
    key = jax.random.PRNGKey(0)
    key, k = jax.random.split(key)
    state = reset_env(env_cfg, k, args.envs)
    history = init_obs_history(env_cfg, state, hp.history_len)

    action = jnp.tile(jnp.array([[0.8, 0.2]], dtype=jnp.float32), (args.envs, 1))
    for _ in range(args.steps):
        key, k = jax.random.split(key)
        state, _, done, _ = step_env(env_cfg, state, action, k)
        frame = fixed_order_policy_obs_from_state(env_cfg, state)
        history = _update_obs_history(history, frame, done > 0.5)

    obs = history.reshape(args.envs, -1)
    node_features, edge_attr, src, dst, _ = _graph_(obs, hp)

    ref_src, ref_dst = build_edge_index(hp.history_len, hp.num_humans)
    assert np.array_equal(np.asarray(src), ref_src) and np.array_equal(np.asarray(dst), ref_dst), \
        "edge_index differs from the numpy template"
    assert node_features.dtype == jnp.float32 and edge_attr.dtype == jnp.float32, \
        f"expected float32, got {node_features.dtype}/{edge_attr.dtype}"
    assert bool(jnp.all(jnp.isfinite(node_features))) and bool(jnp.all(jnp.isfinite(edge_attr)))
    print(f"obs {obs.shape}, nodes {node_features.shape}, edges {edge_attr.shape}: checks passed")

    visualize_graph(
        node_features, edge_attr, src, dst,
        history_len=hp.history_len, num_humans=hp.num_humans,
        b=args.sample, show_masked=args.masked, label_edges=args.labels,
        save_path=args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
