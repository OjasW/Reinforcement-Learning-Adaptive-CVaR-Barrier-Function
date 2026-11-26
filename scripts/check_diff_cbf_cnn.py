"""Checks for model/jax_diff_cbf_cnn.py on real env images. Run from the repo root:
    python scripts/check_diff_cbf_cnn.py

[CBF] New file. Checks (Task 003 step 2):
  1. registry, float32 params with x64 on
  2. pixels only: nominal action and kappa ignore the vector obs, and change when the image changes
  3. the QP does use ground truth: moving a close human changes the safe action, with the images held fixed
  4. QP rows are ordered nearest first, so kappa row j means the j-th nearest human
  5. forward finite; kappa starts at kappa_init; R = r_robot + r_human + margin_base
  6. QP solution respects every CBF row (with its slack) and the box bounds
  7. rollout + one PPO update: finite gradients reach the CNN and kappa head through the QP, kappa logged as "alpha"
  8. deterministic eval runs with images
"""

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
from model import get_jax_model_module, jax_cbf_qp
from trainer.jax_ppo_core import (
    _trainable_params, brax_ppo_loss, compute_brax_gae, init_ppo_opt_state, make_ppo_update_fn,
)
from trainer.jax_ppo_rollout import init_obs_history, make_rollout_collector

failed = []


def ok(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        failed.append(name)


ENVS, STEPS = 16, 32
with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose(config_name="config", overrides=["model=diff_cbf_cnn"])
env_cfg = config_from_omegaconf(cfg)
model = get_jax_model_module(cfg.model)
ok("registry loads model.jax_diff_cbf_cnn", model.__name__ == "model.jax_diff_cbf_cnn", model.__name__)
hp = model.hparams_from_config(cfg.model, cfg.model.critic, act_dim=2)
rc = render_config_from_omegaconf(cfg.model.render)
v_max, w_max = float(cfg.robot.v_max), float(cfg.robot.omega_max)
low, high = np.array([-v_max, -w_max], np.float32), np.array([v_max, w_max], np.float32)
params = model.init_params_from_config(jax.random.PRNGKey(0), cfg.model, cfg.model.critic, 2, low, high)

# ---- 1. dtypes and sizes
ok("all params float32", all(x.dtype == jnp.float32 for x in jax.tree_util.tree_leaves(params)))
n = lambda t: sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(t))
print(f"       params: CNN {n(params['actor']['cnn']):,} | actor {n(params['actor']):,} | critic {n(params['critic']):,}"
      f" | actor keys {sorted(params['actor'])}")

state = reset_env(env_cfg, jax.random.PRNGKey(1), ENVS)
obs_hist = init_obs_history(env_cfg, state, hp.history_len)
obs = obs_hist.reshape((ENVS, -1)).astype(jnp.float32)
img = init_image_history(render_policy_frame(env_cfg, state, rc), hp.image_history_len)
ok("image stack shape", img.shape == (ENVS, hp.image_history_len, 84, 84, 3) and img.dtype == jnp.uint8, str(img.shape))

# ---- 2. the networks see pixels only
# Larger weights than the near-zero init, so a dependence would be visible.
scaled = jax.tree_util.tree_map(lambda x: x, params)
scaled["actor"]["nominal_head"][1] = dict(scaled["actor"]["nominal_head"][1], weight=scaled["actor"]["nominal_head"][1]["weight"] * 100)
scaled["actor"]["kappa_head"] = dict(scaled["actor"]["kappa_head"], weight=scaled["actor"]["kappa_head"]["weight"] * 100)

fwd = lambda p, o, i: model._actor_outputs(p, hp, o, i)
out = fwd(scaled, obs, img)
rand_obs = jax.random.normal(jax.random.PRNGKey(9), obs.shape, dtype=jnp.float32) * 5.0
out_r = fwd(scaled, rand_obs, img)
_n0, nominal_body, kappa = model._network_outputs(scaled, hp, img)
ok("kappa unchanged when obs is randomised", bool(jnp.array_equal(out[3], out_r[3])))
ok("latent unchanged when obs is randomised", bool(jnp.array_equal(out[0], out_r[0])))
# nominal_xy is world frame, so compare it after removing the obs heading (the only obs use)
ok("nominal body action = network output", bool(jnp.allclose(
    model._body_to_world(nominal_body, out[1][:, 4]), out[2], atol=1e-5)))
blank = jnp.zeros_like(img)
_n1, nominal_blank, kappa_blank = model._network_outputs(scaled, hp, blank)
ok("nominal and kappa change with the image",
   float(jnp.max(jnp.abs(nominal_blank - nominal_body))) > 1e-4 and float(jnp.max(jnp.abs(kappa_blank - kappa))) > 1e-4)

# ---- 3. the QP uses ground truth: same images, nearest human moved right in front of the robot
ctx = model.build_qp_context(obs, hp)
qp_obs = ctx["qp_obs"]
theta = qp_obs[:, 4]
ahead = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
moved_qp = qp_obs.at[:, 6:8].set(-0.9 * ahead).at[:, 8:10].set(0.0).at[:, 11].set(1.0)   # rel = robot - human
safe_a = model.safe_action_mean(params, hp, obs, qp_context=ctx, image_obs=img)
safe_b = model.safe_action_mean(params, hp, obs, qp_context={**ctx, "qp_obs": moved_qp}, image_obs=img)
ok("safe action changes with QP geometry, images fixed", float(jnp.max(jnp.abs(safe_a - safe_b))) > 1e-3,
   f"max change {float(jnp.max(jnp.abs(safe_a - safe_b))):.2e}")

# ---- 4. rows are nearest first
blocks = qp_obs[:, 6:].reshape(ENVS, hp.qp_obs_top_k, 6)
dist = np.asarray(jnp.linalg.norm(blocks[:, :, :2], axis=2))
act = np.asarray(blocks[:, :, 5]) > 0.5
sorted_ok = all(np.all(np.diff(dist[b][act[b]]) >= -1e-6) for b in range(ENVS))
ok("QP rows sorted nearest first", sorted_ok)

# ---- 5. forward at init
fwd_j = jax.jit(lambda p, o, i: model._actor_outputs(p, hp, o, i))
latent, qobs, nominal, kap, rs, safe_xy, slack = fwd_j(params, obs, img)
mask = np.asarray(qobs[:, 6:].reshape(ENVS, -1, 6)[:, :, 5]) > 0.5
ok("forward finite, action float32", bool(jnp.all(jnp.isfinite(safe_xy))) and safe_xy.dtype == jnp.float32)
k_act = np.asarray(kap)[mask]
ok("kappa starts near kappa_init", np.allclose(k_act, hp.kappa_init, atol=0.1),
   f"kappa in [{k_act.min():.3f}, {k_act.max():.3f}], init {hp.kappa_init}")
expected_r = float(cfg.robot.radius) + float(cfg.env.humans.radius) + hp.margin_base
ok("R = r_robot + r_human + margin_base", np.allclose(np.asarray(rs)[mask], expected_r, atol=1e-5), f"R = {expected_r:.3f}")

# ---- 6. QP solution feasibility
g, h_qp, _ = jax_cbf_qp._cbf_constraints(hp, qobs.astype(jnp.float64), kap, rs)
viol = jnp.einsum("bkj,bj->bk", g, safe_xy.astype(jnp.float64)) - slack - h_qp
ok("CBF rows hold at the QP solution", float(jnp.max(viol)) < 1e-3, f"max violation {float(jnp.max(viol)):.1e}")
ok("slack >= 0", float(jnp.min(slack)) >= 0.0)
ok("u inside box", bool(jnp.all(safe_xy >= low - 1e-4) & jnp.all(safe_xy <= high + 1e-4)))

# ---- 7. rollout + PPO update
collect = make_rollout_collector(model, use_qp=True, render_cfg=rc)
env_state = (state, obs_hist, img, jnp.zeros((ENVS,), jnp.float32))
batch, _, _ = jax.jit(lambda p, es, k: collect(p, hp, env_cfg, es, k, STEPS))(params, env_state, jax.random.PRNGKey(2))
act_b = np.asarray(batch["act"])
ok("env actions inside bounds", bool(np.all(act_b >= low - 1e-6) & np.all(act_b <= high + 1e-6)))
ok("batch stores uint8 images", batch["img"].dtype == jnp.uint8, str(batch["img"].shape))
ok("kappa logged as alpha", "alpha_active_max" in batch and "beta_active_max" not in batch,
   f"alpha range [{float(batch['alpha_active_min']):.2f}, {float(batch['alpha_active_max']):.2f}]")

adv, ret = compute_brax_gae(batch["rew"], batch["val"], batch["done"], batch["next_value"], gamma=0.99, lam=0.98)
ub = {"obs": batch["obs"].astype(jnp.float32), "img": batch["img"], "act": batch["act"], "policy_act": batch["policy_act"],
      "logp": batch["logp"], "values": batch["val"].reshape(-1), "advantages": adv, "returns": ret,
      "qp_obs": batch["qp_obs"].astype(jnp.float32), "qp_topk_ids": batch["qp_topk_ids"].astype(jnp.int32)}
grads = jax.grad(lambda tp: brax_ppo_loss(tp, params, hp, ub, 0.2, 0.01, 0.5, 0.0, True, model.get_action_and_value)[0])(
    _trainable_params(params))
ok("gradients finite", all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(grads)))


def actor_grad_norm(name_path):
    # policy loss only (vf_coef=0, ent_coef=0), so a non-zero value proves the path through the QP
    g_pi = jax.grad(lambda tp: brax_ppo_loss(tp, params, hp, ub, 0.2, 0.0, 0.0, 0.0, True, model.get_action_and_value)[0])(
        _trainable_params(params))
    x = g_pi["actor"]
    for p in name_path:
        x = x[p]
    return float(jnp.linalg.norm(x["weight"]))


gk = actor_grad_norm(["kappa_head"])
gc = actor_grad_norm(["cnn", "fc"])
ok("policy gradient reaches kappa head through the QP", gk > 0, f"|grad| = {gk:.2e}")
ok("policy gradient reaches the CNN through the QP", gc > 0, f"|grad| = {gc:.2e}")

upd = jax.jit(make_ppo_update_fn(hp, num_epochs=2, num_minibatches=4, max_grad_norm=0.5,
                                 get_action_and_value_fn=model.get_action_and_value))
new_params, _, _, _ = upd(params, init_ppo_opt_state(params), ub, jax.random.PRNGKey(3), 1e-4, 0.2, 0.01, 0.5, 0.5, 0.0)
moved = float(jnp.max(jnp.abs(new_params["actor"]["kappa_head"]["weight"] - params["actor"]["kappa_head"]["weight"])))
finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(new_params))
ok("PPO update runs, params finite, kappa head moves", finite and moved > 0, f"max |d kappa head| = {moved:.2e}")
ok("params stay float32 after update", all(x.dtype == jnp.float32 for x in jax.tree_util.tree_leaves(new_params)))

# ---- 8. deterministic action path used by eval
pa = model.policy_action_mean(params, hp, obs, image_obs=img)
ok("policy_action_mean works without a qp_context", bool(jnp.all(jnp.isfinite(pa))), str(pa.shape))

print("ALL PASS" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)
