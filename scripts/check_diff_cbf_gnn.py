"""Checks for model/jax_diff_cbf_gnn.py and model/jax_cbf_qp.py. Run from the repo root:
    python scripts/check_diff_cbf_gnn.py

[CBF] New file. Checks (Task 003 step 1):
  1. registry, float32 params with x64 on, parameter counts
  2. CBF row is correct: G u - h_qp == -(hdot + kappa h), with hdot from finite differences of the real motion
  3. CBF row equals the CVaR row once the CVaR terms are switched off (one GMM mean, zero variance), up to the
     known logsumexp offset 0.1*ln(3) and the 1e-4 sigma floor
  4. inactive slots give all-zero rows
  5. real env obs: jit forward finite; kappa starts at kappa_init; R = r_robot + r_human + margin_base
  6. QP solution respects every CBF row (with its slack), slack >= 0, and the box bounds
  7. kappa logits are equivariant to human-slot permutations
  8. rollout + one PPO update: finite gradients reach the kappa head and the GAT, kappa is logged as "alpha"
"""

import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax

jax.config.update("jax_enable_x64", True)  # same as trainer/jax_platform.py: catches float64 leaks
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("add", lambda *v: sum(v), replace=True)
OmegaConf.register_new_resolver("mul", lambda *v: math.prod(v), replace=True)

from env.jax_social_nav import config_from_omegaconf, reset_env
from model import get_jax_model_module, jax_cbf_qp, jax_cvar_qp
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
    cfg = compose(config_name="config", overrides=["model=diff_cbf_gnn"])
env_cfg = config_from_omegaconf(cfg)
model = get_jax_model_module(cfg.model)
ok("registry loads model.jax_diff_cbf_gnn", model.__name__ == "model.jax_diff_cbf_gnn", model.__name__)
hp = model.hparams_from_config(cfg.model, cfg.model.critic, act_dim=2)
v_max, w_max = float(cfg.robot.v_max), float(cfg.robot.omega_max)
low, high = np.array([-v_max, -w_max], np.float32), np.array([v_max, w_max], np.float32)
params = model.init_params_from_config(jax.random.PRNGKey(0), cfg.model, cfg.model.critic, 2, low, high)

# ---- 1. dtypes and sizes
ok("all params float32", all(x.dtype == jnp.float32 for x in jax.tree_util.tree_leaves(params)))
n = lambda t: sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(t))
print(f"       params: GAT {n(params['actor']['gat']):,} | actor {n(params['actor']):,} | critic {n(params['critic']):,}"
      f" | total {n(params):,} | actor keys {sorted(k for k in params['actor'] if k != 'gat')}")


# ---- 2. one CBF row against finite differences of the true motion
def obs_one(robot_xy, theta, human_xy, human_v, r_h=0.4, r_r=0.3, active=1.0):
    """One-human QP obs row in env convention: rel = robot - human."""
    rel = np.asarray(robot_xy) - np.asarray(human_xy)
    return np.array([[0, 0, 0, 0, theta, r_r, *rel, *human_v, r_h, active]], dtype=np.float64)


def h_of(robot_xy, theta, human_xy, R, eps):
    q = np.asarray(robot_xy) + eps * np.array([math.cos(theta), math.sin(theta)])
    d = q - np.asarray(human_xy)
    return float(d @ d - (R + eps) ** 2)


rng = np.random.default_rng(0)
eps, dt, worst = hp.lookahead_distance, 1e-6, 0.0
for _ in range(20):
    robot, human = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
    theta, hv, u = rng.uniform(-math.pi, math.pi), rng.uniform(-1, 1, 2), rng.uniform(-1.5, 1.5, 2)
    kappa, R = rng.uniform(0.5, 5.0), 0.75
    obs = jnp.asarray(obs_one(robot, theta, human, hv))
    g, h_qp, _ = jax_cbf_qp._cbf_constraints(hp, obs, jnp.array([[kappa]]), jnp.array([[R]]))
    row = float(g[0, 0] @ u - h_qp[0, 0])
    robot2 = robot + dt * u[0] * np.array([math.cos(theta), math.sin(theta)])
    h0 = h_of(robot, theta, human, R, eps)
    h1 = h_of(robot2, theta + dt * u[1], human + dt * hv, R, eps)
    expect = -((h1 - h0) / dt + kappa * h0)
    worst = max(worst, abs(row - expect) / (1.0 + abs(expect)))
ok("G u - h_qp == -(hdot + kappa h) (finite differences)", worst < 1e-4, f"max rel err {worst:.1e}")

# ---- 3. equals the CVaR row with the CVaR terms off
K = 5
blocks = np.concatenate([rng.uniform(-4, 4, (8, K, 2)), rng.uniform(-1, 1, (8, K, 2)),
                         np.full((8, K, 1), 0.4), np.ones((8, K, 1))], axis=-1)
head = np.concatenate([np.zeros((8, 4)), rng.uniform(-3, 3, (8, 1)), np.full((8, 1), 0.3)], axis=1)
qp_obs = jnp.asarray(np.concatenate([head, blocks.reshape(8, -1)], axis=1))
r_safe = jnp.full((8, K), 0.75)
cvar_hp = SimpleNamespace(**{**hp.__dict__, "alpha": 2.0, "gmm_lateral_ratio": 0.0})
gmm = {"gmm_weights": jnp.array([0.6, 0.2, 0.2]), "gmm_variances": jnp.zeros(3)}
beta = jnp.full((8, K), 0.1)
g_cv, h_cv, _ = jax_cvar_qp._unicycle_cbf_constraints(gmm, cvar_hp, qp_obs, beta, r_safe)
g_cb, h_cb, _ = jax_cbf_qp._unicycle_cbf_constraints(hp, qp_obs, jnp.full((8, K), 2.0), r_safe)
offset = 0.1 * math.log(3.0) + 1e-4 * float(jax_cvar_qp._cvar_coeff_from_beta(jnp.array(0.1)))
ok("G rows identical to CVaR", bool(jnp.allclose(g_cv, g_cb, atol=1e-10)))
ok("h rows = CVaR rows + known offset", bool(jnp.allclose(h_cb - h_cv, offset, atol=1e-8)), f"offset {offset:.4f}")

# ---- 4. masked slots
masked = qp_obs.at[:, 6 + 5::6].set(0.0)
g_m, h_m, _ = jax_cbf_qp._cbf_constraints(hp, masked, jnp.full((8, K), 2.0), r_safe)
ok("inactive slots give zero rows", float(jnp.max(jnp.abs(g_m))) == 0.0 and float(jnp.max(jnp.abs(h_m))) == 0.0)

# ---- 5. real env forward
state = reset_env(env_cfg, jax.random.PRNGKey(1), ENVS)
obs_hist = init_obs_history(env_cfg, state, hp.history_len)
obs = obs_hist.reshape((ENVS, -1)).astype(jnp.float32)
fwd = jax.jit(lambda p, o: model._actor_outputs(p, hp, o))
latent, qobs, nominal, kappa, rs, safe_xy, slack = fwd(params, obs)
mask = np.asarray(qobs[:, 6:].reshape(ENVS, -1, 6)[:, :, 5]) > 0.5
ok("forward finite, action float32", bool(jnp.all(jnp.isfinite(safe_xy))) and safe_xy.dtype == jnp.float32)
k_act = np.asarray(kappa)[mask]
ok("kappa starts near kappa_init", np.allclose(k_act, hp.kappa_init, atol=0.05),
   f"kappa in [{k_act.min():.3f}, {k_act.max():.3f}], init {hp.kappa_init}")
ok("R = r_robot + r_human + margin_base", np.allclose(np.asarray(rs)[mask], 0.3 + 0.4 + hp.margin_base, atol=1e-6),
   f"R = {float(np.asarray(rs)[mask][0]):.3f}")

# ---- 6. QP solution feasibility
g, h_qp, _ = jax_cbf_qp._cbf_constraints(hp, qobs.astype(jnp.float64), kappa, rs)
viol = jnp.einsum("bkj,bj->bk", g, safe_xy.astype(jnp.float64)) - slack - h_qp
ok("CBF rows hold at the QP solution", float(jnp.max(viol)) < 1e-3, f"max violation {float(jnp.max(viol)):.1e}")
ok("slack >= 0", float(jnp.min(slack)) >= 0.0)

# same check where constraints are active: synthetic close humans, nominal aimed at the nearest one
close = np.array(blocks)
close[:, :, 0:2] *= 0.35                                                    # humans 0 to 1.4 m away
cobs = jnp.asarray(np.concatenate([head, close.reshape(8, -1)], axis=1))
nearest = np.argmin(np.linalg.norm(close[:, :, 0:2], axis=2), axis=1)
toward = -close[np.arange(8), nearest, 0:2]                                 # rel = robot - human
nom = jnp.asarray(1.5 * toward / np.linalg.norm(toward, axis=1, keepdims=True))
kap = jnp.full((8, K), 2.0)
u_c, s_c = jax_cbf_qp.solve_slack_qp(params, hp, cobs, nom, kap, r_safe)
g_c, h_c, _ = jax_cbf_qp._cbf_constraints(hp, cobs, kap, r_safe)
viol_c = jnp.einsum("bkj,bj->bk", g_c, u_c) - s_c - h_c
active = int(jnp.sum(viol_c > -1e-3))
ok("CBF rows hold when active", float(jnp.max(viol_c)) < 1e-3 and active > 0,
   f"{active} active rows, max violation {float(jnp.max(viol_c)):.1e}, max slack {float(jnp.max(s_c)):.2f}")
ok("u inside box", bool(jnp.all(safe_xy >= low - 1e-4) & jnp.all(safe_xy <= high + 1e-4)))

# ---- 7. slot permutation equivariance of kappa logits
perm = np.random.default_rng(3).permutation(hp.num_humans)
frames = obs.reshape(ENVS, hp.history_len, 6 + 6 * hp.num_humans)
hb = frames[:, :, 6:].reshape(ENVS, hp.history_len, hp.num_humans, 6)
obs_p = jnp.concatenate([frames[:, :, :6], hb[:, :, perm].reshape(ENVS, hp.history_len, -1)], -1).reshape(ENVS, -1)


def logits_all(p, o):
    _lat, human_in = model._encode_with_humans(p, hp, o)
    x = model.activation(model.dense(p["actor"]["human_fc1"], human_in), hp.actor_act)
    return model.dense(p["actor"]["human_kappa_head"], x)[..., 0]


la, lp = logits_all(params, obs), logits_all(params, obs_p)
err = float(jnp.max(jnp.abs(la[:, perm] - lp)))
ok("kappa logits slot-permutation equivariant", err < 1e-5, f"max err {err:.1e}")

# ---- 8. rollout + PPO update
collect = make_rollout_collector(model, use_qp=True)
env_state = (state, obs_hist, jnp.zeros((ENVS,), jnp.float32))
batch, _, _ = jax.jit(lambda p, es, k: collect(p, hp, env_cfg, es, k, STEPS))(params, env_state, jax.random.PRNGKey(2))
act = np.asarray(batch["act"])
ok("env actions inside bounds", bool(np.all(act >= low - 1e-6) & np.all(act <= high + 1e-6)))
ok("kappa logged as alpha", "alpha_active_max" in batch and "beta_active_max" not in batch,
   f"alpha range [{float(batch['alpha_active_min']):.2f}, {float(batch['alpha_active_max']):.2f}]")

adv, ret = compute_brax_gae(batch["rew"], batch["val"], batch["done"], batch["next_value"], gamma=0.99, lam=0.98)
ub = {"obs": batch["obs"].astype(jnp.float32), "act": batch["act"], "policy_act": batch["policy_act"],
      "logp": batch["logp"], "values": batch["val"].reshape(-1), "advantages": adv, "returns": ret,
      "qp_obs": batch["qp_obs"].astype(jnp.float32), "qp_topk_ids": batch["qp_topk_ids"].astype(jnp.int32)}
grads = jax.grad(lambda tp: brax_ppo_loss(tp, params, hp, ub, 0.2, 0.01, 0.5, 0.0, True, model.get_action_and_value)[0])(
    _trainable_params(params))
ok("gradients finite", all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(grads)))
gk = float(jnp.linalg.norm(grads["actor"]["human_kappa_head"]["weight"]))
gg = float(jnp.linalg.norm(grads["actor"]["gat"]["node_enc"]["weight"]))
ok("gradient reaches kappa head through the QP", gk > 0, f"|grad kappa head| = {gk:.2e}")
ok("gradient reaches GAT", gg > 0, f"|grad GAT node_enc| = {gg:.2e}")

upd = jax.jit(make_ppo_update_fn(hp, num_epochs=2, num_minibatches=4, max_grad_norm=0.5,
                                 get_action_and_value_fn=model.get_action_and_value))
new_params, _, _, _ = upd(params, init_ppo_opt_state(params), ub, jax.random.PRNGKey(3), 1e-4, 0.2, 0.01, 0.5, 0.5, 0.0)
moved = float(jnp.max(jnp.abs(new_params["actor"]["human_kappa_head"]["weight"] - params["actor"]["human_kappa_head"]["weight"])))
finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(new_params))
ok("PPO update runs, params finite, kappa head moves", finite and moved > 0, f"max |d kappa head| = {moved:.2e}")
ok("params stay float32 after update", all(x.dtype == jnp.float32 for x in jax.tree_util.tree_leaves(new_params)))

print("ALL PASS" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)
