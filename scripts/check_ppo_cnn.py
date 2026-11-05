"""Checks for model/jax_ppo_cnn.py on real env images. Run from the repo root:
    python scripts/check_ppo_cnn.py                 # critic.input=image (default)
    python scripts/check_ppo_cnn.py --critic state  # asymmetric critic

[IMG] New file. Checks (Task 002 step 3):
  1. layer shapes / parameter counts, all float32 even with x64 on (as in the trainer)
  2. actor is pixels-only: changing the vector obs leaves the action mean unchanged; changing the images changes it
  3. sampled env actions stay inside [action_low, action_high]
  4. one full PPO update (rollout -> GAE -> minibatch epochs) runs, with finite gradients that reach the CNN
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax

jax.config.update("jax_enable_x64", True)  # same as trainer/jax_platform.py: catches float64 leaks
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("add", lambda *v: sum(v), replace=True)
OmegaConf.register_new_resolver("mul", lambda *v: math.prod(v), replace=True)

from env.jax_image_renderer import init_image_history, render_config_from_omegaconf, render_policy_frame
from env.jax_social_nav import config_from_omegaconf, reset_env
from model import get_jax_model_module
from trainer.jax_ppo_core import (
    _trainable_params, brax_ppo_loss, compute_brax_gae, init_ppo_opt_state, make_ppo_update_fn,
)
from trainer.jax_ppo_rollout import init_obs_history, make_rollout_collector

failed = []


def ok(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        failed.append(name)


ap = argparse.ArgumentParser()
ap.add_argument("--critic", default="image", choices=["image", "state"])
ap.add_argument("--envs", type=int, default=16)
ap.add_argument("--steps", type=int, default=32)
args = ap.parse_args()

with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose(config_name="config", overrides=["model=ppo_cnn", f"model.critic.input={args.critic}"])
env_cfg = config_from_omegaconf(cfg)
model = get_jax_model_module(cfg.model)
ok("registry loads model.jax_ppo_cnn", model.__name__ == "model.jax_ppo_cnn", model.__name__)
hp = model.hparams_from_config(cfg.model, cfg.model.critic, act_dim=2)
rc = render_config_from_omegaconf(cfg.model.render)
v_max, w_max = float(cfg.robot.v_max), float(cfg.robot.omega_max)
low, high = np.array([-v_max, -w_max], np.float32), np.array([v_max, w_max], np.float32)
params = model.init_params_from_config(jax.random.PRNGKey(0), cfg.model, cfg.model.critic, 2, low, high)

# ---- 1. shapes, sizes, dtypes
leaves = jax.tree_util.tree_leaves(params)
ok("all params float32", all(x.dtype == jnp.float32 for x in leaves))
n = lambda t: sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(t))
print(f"       input {hp.image_size}x{hp.image_size}x{hp.in_channels} -> conv out {hp.conv_out_hw}x{hp.conv_out_hw}"
      f"x{hp.cnn_channels[-1]} = {hp.flat_dim} -> latent {hp.cnn_latent_dim}")
for i, layer in enumerate(params["actor"]["cnn"]["conv"]):
    print(f"       conv{i + 1}: kernel {tuple(layer['weight'].shape)} (HWIO), stride {hp.cnn_strides[i]}")
print(f"       params: CNN {n(params['actor']['cnn']):,} | actor head {n(params['actor']) - n(params['actor']['cnn']):,}"
      f" | critic {n(params['critic']):,} | total {n(params):,}")

state = reset_env(env_cfg, jax.random.PRNGKey(1), args.envs)
obs_hist = init_obs_history(env_cfg, state, hp.history_len)
img_hist = init_image_history(render_policy_frame(env_cfg, state, rc), hp.image_history_len)
obs = obs_hist.reshape((args.envs, -1))
mean = jax.jit(lambda p, o, im: model.policy_action_mean(p, hp, o, image_obs=im))
value = jax.jit(lambda p, o, im: model.critic_value(p, hp, o, image_obs=im))
m0, v0 = mean(params, obs, img_hist), value(params, obs, img_hist)
ok("outputs float32 and finite", m0.dtype == jnp.float32 and v0.dtype == jnp.float32
   and bool(jnp.all(jnp.isfinite(m0)) & jnp.all(jnp.isfinite(v0))), f"mean {m0.shape} value {v0.shape}")

# ---- 2. actor sees pixels only
obs_noise = obs + 5.0 * jax.random.normal(jax.random.PRNGKey(9), obs.shape, dtype=jnp.float32)
m_noise, v_noise = mean(params, obs_noise, img_hist), value(params, obs_noise, img_hist)
ok("actor ignores vector obs", bool(jnp.array_equal(m0, m_noise)))
img_shift = jnp.roll(img_hist, 6, axis=3)                     # move everything 6 px sideways
ok("actor responds to images", float(jnp.max(jnp.abs(mean(params, obs, img_shift) - m0))) > 0)
if hp.critic_input == "image":
    ok("critic (image) ignores vector obs", bool(jnp.array_equal(v0, v_noise)))
else:
    ok("critic (state) uses vector obs, actor still does not", not bool(jnp.array_equal(v0, v_noise)))
try:
    model.policy_action_mean(params, hp, obs)
    ok("clear error without images", False)
except ValueError as e:
    ok("clear error without images", "image_obs" in str(e))

# ---- 3 + 4. rollout (actions), then one PPO update
env_state = (state, obs_hist, img_hist, jnp.zeros((args.envs,), jnp.float32))
collect = make_rollout_collector(model, use_qp=False, render_cfg=rc)
batch, _, _ = jax.jit(lambda p, es, k: collect(p, hp, env_cfg, es, k, args.steps))(params, env_state, jax.random.PRNGKey(2))
act = np.asarray(batch["act"])
ok("env actions inside bounds", bool(np.all(act >= low - 1e-6) & np.all(act <= high + 1e-6)),
   f"v in [{act[:, 0].min():.2f},{act[:, 0].max():.2f}] (limit {v_max}), "
   f"w in [{act[:, 1].min():.2f},{act[:, 1].max():.2f}] (limit {w_max})")

adv, ret = compute_brax_gae(batch["rew"], batch["val"], batch["done"], batch["next_value"], gamma=0.99, lam=0.98)
ub = {"obs": batch["obs"].astype(jnp.float32), "act": batch["act"], "policy_act": batch["policy_act"],
      "logp": batch["logp"], "values": batch["val"].reshape(-1), "advantages": adv, "returns": ret,
      "img": batch["img"]}
grads = jax.grad(lambda tp: brax_ppo_loss(tp, params, hp, ub, 0.2, 0.01, 0.5, 0.0, True, model.get_action_and_value)[0])(
    _trainable_params(params))
g_leaves = jax.tree_util.tree_leaves(grads)
ok("gradients finite", all(bool(jnp.all(jnp.isfinite(g))) for g in g_leaves))
conv1_g = float(jnp.linalg.norm(grads["actor"]["cnn"]["conv"][0]["weight"]))
ok("gradient reaches first conv layer", conv1_g > 0, f"|grad conv1| = {conv1_g:.2e}")

upd = jax.jit(make_ppo_update_fn(hp, num_epochs=2, num_minibatches=4, max_grad_norm=0.5,
                                 get_action_and_value_fn=model.get_action_and_value))
new_params, _, _, metrics = upd(params, init_ppo_opt_state(params), ub, jax.random.PRNGKey(3),
                                1e-4, 0.2, 0.01, 0.5, 0.5, 0.0)
moved = float(jnp.max(jnp.abs(new_params["actor"]["cnn"]["conv"][0]["weight"] - params["actor"]["cnn"]["conv"][0]["weight"])))
finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(new_params))
ok("PPO update runs, params finite and CNN updated", finite and moved > 0, f"max |d conv1| = {moved:.2e}")
ok("params stay float32 after update", all(x.dtype == jnp.float32 for x in jax.tree_util.tree_leaves(new_params)))

print("ALL PASS" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)