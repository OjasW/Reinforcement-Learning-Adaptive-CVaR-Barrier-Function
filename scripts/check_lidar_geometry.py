"""Checks for the simulated LiDAR, the circle fit and the tracker (Task 003 step 3).  # [LIDAR]

Run from the repo root:  python scripts/check_lidar_geometry.py
Each line prints PASS/FAIL; the script exits non-zero if anything fails. Section 9 also prints
perception errors on real env rollouts (no thresholds: they are numbers to report and tune against).
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax

jax.config.update("jax_enable_x64", True)  # same as the trainer: catches float64 leaks
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("add", lambda *v: sum(v), replace=True)  # same as the other check scripts
OmegaConf.register_new_resolver("mul", lambda *v: math.prod(v), replace=True)

from env import jax_lidar as L
from env import jax_tracker as T
from env.jax_social_nav import config_from_omegaconf, fixed_order_policy_obs_from_state, reset_env, step_env

failed = []


def ok(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        failed.append(name)


with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose(config_name="config", overrides=["model=diff_cbf_cnn"])
env_cfg = config_from_omegaconf(cfg)
lcfg = L.lidar_config_from_omegaconf(None, env_cfg)
tcfg = T.tracker_config_from_omegaconf(None, env_cfg)
R = lcfg.known_radius
print(f"lidar: {lcfg.num_beams} beams, fov {math.degrees(lcfg.fov):.0f} deg, range {lcfg.max_range} m, "
      f"radius {R} m | tracker: {tcfg.num_slots} slots, dt {tcfg.dt}, gate {tcfg.gate} m")


def one_env(robot_xy, theta, humans, mask=None, radius=R):
    humans = jnp.asarray(humans, jnp.float32)[None]
    H = humans.shape[1]
    mask = jnp.ones((1, H), jnp.float32) if mask is None else jnp.asarray(mask, jnp.float32)[None]
    return (jnp.asarray(robot_xy, jnp.float32)[None], jnp.asarray([theta], jnp.float32), humans,
            jnp.full((1, H), radius, jnp.float32), mask)


# ---- 1. ray hits match a brute-force reference (float64, vectorised)
# Beams that graze a circle's edge are skipped: there |r^2 - perp^2| ~ 0, so float32 (scan) and float64
# (reference) can disagree on hit vs miss, giving an error of up to max_range on that single beam.
state = reset_env(env_cfg, jax.random.PRNGKey(0), 8)
scan = L.simulate_scan(lcfg, state.robot_state[:, :2], state.robot_state[:, 2], state.human_positions,
                       state.human_radii, state.human_mask)
ang = np.asarray(L.beam_angles(lcfg), dtype=np.float64)
rs, hp_, hr, hm = (np.asarray(x, dtype=np.float64) for x in
                   (state.robot_state, state.human_positions, state.human_radii, state.human_mask))
a = rs[:, 2:3] + ang[None, :]
dirs = np.stack([np.cos(a), np.sin(a)], axis=-1)  # (B, N, 2)
oc = hp_ - rs[:, None, :2]  # (B, H, 2)
t_ca = np.einsum("bnk,bhk->bnh", dirs, oc)
disc = hr[:, None, :] ** 2 - ((oc ** 2).sum(-1)[:, None, :] - t_ca ** 2)
t_hit = t_ca - np.sqrt(np.maximum(disc, 0.0))
hit = (disc >= 0) & (t_hit > 0) & (hm[:, None, :] > 0.5)
ref = np.minimum(np.where(hit, t_hit, np.inf).min(-1), lcfg.max_range)
grazing = np.any((np.abs(disc) < 1e-3) & (hm[:, None, :] > 0.5) & (t_ca > 0), axis=-1)  # within ~1 mm of an edge
err = np.abs(np.asarray(scan.ranges, dtype=np.float64) - ref)
ok("ranges match brute force", err[~grazing].max() < 1e-3,
   f"max err {err[~grazing].max():.1e} m on {int((~grazing).sum())} beams ({int(grazing.sum())} grazing beams skipped, "
   f"max err there {err[grazing].max() if grazing.any() else 0.0:.1e} m)")
ok("scan is float32", scan.ranges.dtype == jnp.float32 and scan.points.dtype == jnp.float32)

# ---- 2. inactive humans give no hits
mask = state.human_mask.at[:, ::2].set(0.0)
scan_m = L.simulate_scan(lcfg, state.robot_state[:, :2], state.robot_state[:, 2], state.human_positions,
                         state.human_radii, mask)
hit_ids = np.asarray(scan_m.hit_ids)
ok("masked humans never hit", not np.any((hit_ids >= 0) & (hit_ids % 2 == 0)))

# ---- 3. occlusion: a human straight behind another is not detected
dets, _, _ = L.detect_obstacles(lcfg, *one_env([0, 0], 0.0, [[2.0, 0.0], [4.0, 0.0]]))
c = np.asarray(dets.centers[0])[np.asarray(dets.valid[0])]
ok("hidden human not detected", len(c) == 1 and np.linalg.norm(c[0] - [2.0, 0.0]) < 0.05, f"detections {c.round(3).tolist()}")

# ---- 4. a human on the -pi/pi seam (directly behind) is one cluster, not two
dets, scan1, labels = L.detect_obstacles(lcfg, *one_env([0, 0], 0.0, [[-3.0, 0.0]]))
nclu = int(np.sum(np.asarray(dets.valid[0])))
ok("cluster across the beam seam is merged", nclu == 1 and np.linalg.norm(np.asarray(dets.centers[0, 0]) - [-3, 0]) < 0.05,
   f"{nclu} cluster(s)")

# ---- 5. two separated humans -> two clusters; touching humans -> one (known limitation, counted later)
d2, _, _ = L.detect_obstacles(lcfg, *one_env([0, 0], 0.0, [[3.0, -0.6], [3.0, 0.6]]))
d3, _, _ = L.detect_obstacles(lcfg, *one_env([0, 0], 0.0, [[3.0, -0.41], [3.0, 0.41]]))
ok("humans 0.4 m apart are two clusters", int(np.sum(np.asarray(d2.valid[0]))) == 2)
print(f"       humans 0.02 m apart -> {int(np.sum(np.asarray(d3.valid[0])))} cluster(s) (the range jump at the shared edge splits them)")

# ---- 6. circle fit vs centroid at several ranges, with 2 cm noise
noisy = lcfg._replace(range_noise_std=0.02)
rng = np.random.default_rng(0)
rows = []
for rng_m in (1.0, 3.0, 6.0, 10.0):
    B = 256
    a = rng.uniform(-np.pi, np.pi, B)
    hum = np.stack([rng_m * np.cos(a), rng_m * np.sin(a)], 1)[:, None, :].astype(np.float32)
    d, _, _ = L.detect_obstacles(noisy, jnp.zeros((B, 2)), jnp.zeros((B,)), jnp.asarray(hum),
                                 jnp.full((B, 1), R), jnp.ones((B, 1)), jax.random.PRNGKey(int(rng_m * 10)))
    v = np.asarray(d.valid)
    first = np.argmax(v, axis=1)
    cen = np.asarray(d.centers)[np.arange(B), first]
    mean = np.asarray(d.means)[np.arange(B), first]
    rows.append((rng_m, np.linalg.norm(cen - hum[:, 0], axis=1).mean(), np.linalg.norm(mean - hum[:, 0], axis=1).mean(),
                 np.asarray(d.counts)[np.arange(B), first].mean()))
for r_, ef, em, cnt in rows:
    print(f"       range {r_:4.1f} m: fit {ef:.3f} m | centroid {em:.3f} m | {cnt:.1f} hits")
ok("circle fit error < 0.05 m (the safety margin) at every range", all(r[1] < 0.05 for r in rows))
ok("fit beats the centroid by > 5x", all(r[2] > 5 * r[1] for r in rows))

# ---- 7. tracker: velocity in the world frame while the robot drives and turns
steps = 40
robot, th = np.array([-3.0, 0.0]), 0.0
hum, hv = np.array([[3.0, -2.0]]), np.array([[-0.6, 0.8]])
tr = T.init_tracker(tcfg, 1)
step_j = jax.jit(lambda t, d, r: T.tracker_step(tcfg, t, d, r))
for k in range(steps):
    d, _, _ = L.detect_obstacles(noisy, *one_env(robot, th, hum), key=jax.random.PRNGKey(100 + k))
    tr = step_j(tr, d, jnp.asarray([k == 0]))
    hum = hum + hv * tcfg.dt
    robot = robot + 1.0 * tcfg.dt * np.array([math.cos(th), math.sin(th)])
    th += 0.4 * tcfg.dt
s = int(np.argmax(np.asarray(tr.active[0])))
v_est = np.asarray(tr.x[0, s, 2:4])
ok("velocity correct under ego-motion (error < 0.15 m/s)", np.linalg.norm(v_est - hv[0]) < 0.15,
   f"est {v_est.round(3).tolist()} true {hv[0].tolist()}")
ok("one track only", int(np.sum(np.asarray(tr.active))) == 1)

# ---- 8. reset clears tracks; a lost human coasts, then is dropped
tr2 = step_j(tr, L.Detections(jnp.zeros((1, lcfg.max_clusters, 2)), jnp.zeros((1, lcfg.max_clusters), bool),
                              jnp.zeros((1, lcfg.max_clusters)), jnp.zeros((1, lcfg.max_clusters, 2))), jnp.asarray([True]))
ok("reset clears all tracks", int(np.sum(np.asarray(tr2.active))) == 0)
empty = L.Detections(jnp.zeros((1, lcfg.max_clusters, 2)), jnp.zeros((1, lcfg.max_clusters), bool),
                     jnp.zeros((1, lcfg.max_clusters)), jnp.zeros((1, lcfg.max_clusters, 2)))
tr3 = tr
alive = []
for k in range(tcfg.max_coast + 2):
    tr3 = step_j(tr3, empty, jnp.asarray([False]))
    alive.append(int(np.sum(np.asarray(tr3.active))))
ok(f"unmatched track coasts {tcfg.max_coast} steps then drops", alive[tcfg.max_coast - 1] == 1 and alive[-1] == 0, str(alive))

# ---- 9. real env: frame layout, no leakage, perception error over a rollout
B, STEPS = 64, 150
state = reset_env(env_cfg, jax.random.PRNGKey(3), B)


@jax.jit
def run(state, key):
    def body(carry, k):
        state, tr, first = carry
        k_scan, k_env = jax.random.split(k)
        d, _, _ = L.detect_from_state(noisy, state, k_scan)
        tr = T.tracker_step(tcfg, tr, d, jnp.full((B,), first))
        m = T.perception_metrics(state, tr, tcfg, near_range=4.0)
        t = state.step_count.astype(jnp.float32) * env_cfg.dt
        action = jnp.stack([jnp.full((B,), 0.6), 0.8 * jnp.sin(0.7 * t + jnp.arange(B))], 1)
        state, _, _, _ = step_env(env_cfg, state, action, k_env)
        return (state, tr, False), m
    keys = jax.random.split(key, STEPS)
    (state, tr, _), ms = jax.lax.scan(body, (state, T.init_tracker(tcfg, B), True), keys)
    return state, tr, ms


t0 = time.time()
_, _, _ = jax.block_until_ready(run(state, jax.random.PRNGKey(4)))
t1 = time.time()
end_state, tr, ms = jax.block_until_ready(run(state, jax.random.PRNGKey(5)))
t2 = time.time()
print(f"       {B} envs x {STEPS} steps: compile+run {t1 - t0:.1f} s, run {t2 - t1:.2f} s (CPU)")
frame = T.lidar_policy_obs(env_cfg, end_state, tr, tcfg)
gt = fixed_order_policy_obs_from_state(env_cfg, end_state)
ok("frame has the ground-truth layout and dtype", frame.shape == gt.shape and frame.dtype == jnp.float32, str(frame.shape))
ok("robot block identical to ground truth", bool(jnp.array_equal(frame[:, :6], gt[:, :6])))
leak = end_state._replace(human_vels=end_state.human_vels + 5.0, human_positions=end_state.human_positions + 3.0,
                          human_mask=1.0 - end_state.human_mask)
ok("no leakage: frame unchanged when true human state changes", bool(jnp.array_equal(frame, T.lidar_policy_obs(env_cfg, leak, tr, tcfg))))
late = slice(20, None)  # skip the first 2 s while velocities converge
pe = np.asarray(ms["pos_err"])[late]
ve = np.asarray(ms["vel_err"])[late]
nm = np.asarray(ms["num_matched"])[late]
w = nm / max(nm.sum(), 1)
print(f"       humans within 4 m: pos err {np.sum(pe * w):.3f} m | vel err {np.sum(ve * w):.3f} m/s | "
      f"missed {np.asarray(ms['missed'])[late].mean():.2f}/env | false tracks {np.asarray(ms['false_tracks'])[late].mean():.2f}/env")
ok("perception metrics finite", all(np.all(np.isfinite(np.asarray(v))) for v in ms.values()))

# ---- 10. diff_cbf_cnn end to end with model.geometry=lidar: rollout, stored QP context, PPO update, eval
from env.jax_image_renderer import init_image_history, render_config_from_omegaconf, render_policy_frame
from model import get_jax_model_module
from trainer.jax_ppo_core import (
    _trainable_params, brax_ppo_loss, compute_brax_gae, init_ppo_opt_state, make_ppo_update_fn,
)
from trainer.jax_ppo_eval import evaluate_jax_env
from trainer.jax_ppo_rollout import make_rollout_collector

with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg_l = compose(config_name="config", overrides=["model=diff_cbf_cnn", "model.geometry=lidar",
                                                     "model.lidar.range_noise_std=0.02", "model.lidar.dropout_prob=0.01"])
geo = T.geometry_from_omegaconf(cfg_l.model, env_cfg)
ok("config builds a LidarGeometry", geo is not None and geo.source == "lidar" and geo.lidar.range_noise_std == 0.02)
model = get_jax_model_module(cfg_l.model)
hp = model.hparams_from_config(cfg_l.model, cfg_l.model.critic, act_dim=2)
rc = render_config_from_omegaconf(cfg_l.model.render)
v_max, w_max = float(cfg_l.robot.v_max), float(cfg_l.robot.omega_max)
low, high = np.array([-v_max, -w_max], np.float32), np.array([v_max, w_max], np.float32)
params = model.init_params_from_config(jax.random.PRNGKey(0), cfg_l.model, cfg_l.model.critic, 2, low, high)
ENVS, STEPS_R = 16, 48
st = reset_env(env_cfg, jax.random.PRNGKey(11), ENVS)
hist, tr0 = T.init_lidar_obs_history(env_cfg, geo, st, hp.history_len, jax.random.PRNGKey(11))
img = init_image_history(render_policy_frame(env_cfg, st, rc), hp.image_history_len)
env_state = (st, hist, img, tr0, jnp.zeros((ENVS,), jnp.float32))
collect = make_rollout_collector(model, use_qp=True, render_cfg=rc, geometry=geo)
batch, next_env_state, _ = jax.jit(lambda p, es, k: collect(p, hp, env_cfg, es, k, STEPS_R))(params, env_state, jax.random.PRNGKey(12))
ok("rollout returns a 5-tuple env_state with the tracker", len(next_env_state) == 5 and isinstance(next_env_state[3], T.TrackerState))
ok("perception summary in batch", "lidar_pos_err_sum" in batch,
   f"pos err {float(batch['lidar_pos_err_sum'] / max(float(batch['lidar_pos_err_count']), 1)):.3f} m, "
   f"vel err {float(batch['lidar_vel_err_sum'] / max(float(batch['lidar_vel_err_count']), 1)):.3f} m/s")
ctx = model.build_qp_context(batch["obs"].astype(jnp.float32), hp)
ok("stored qp_obs = QP context rebuilt from the stored (LiDAR) obs", bool(jnp.array_equal(ctx["qp_obs"], batch["qp_obs"])))
_a, logp_re, _e, _v = model.get_action_and_value(
    params, hp, batch["obs"].astype(jnp.float32), policy_action=batch["policy_act"],
    qp_context={"qp_obs": batch["qp_obs"], "qp_topk_ids": batch["qp_topk_ids"]}, image_obs=batch["img"])[:4]
ok("log-prob recomputed from the stored context matches the rollout", float(jnp.max(jnp.abs(logp_re - batch["logp"]))) < 1e-4,
   f"max diff {float(jnp.max(jnp.abs(logp_re - batch['logp']))):.1e}")
gt_frames = fixed_order_policy_obs_from_state(env_cfg, next_env_state[0])
ok("tracker frames are not the ground-truth frames", not bool(jnp.allclose(next_env_state[1][:, -1], gt_frames)))
adv, ret = compute_brax_gae(batch["rew"], batch["val"], batch["done"], batch["next_value"], gamma=0.99, lam=0.98)
ub = {"obs": batch["obs"].astype(jnp.float32), "img": batch["img"], "act": batch["act"], "policy_act": batch["policy_act"],
      "logp": batch["logp"], "values": batch["val"].reshape(-1), "advantages": adv, "returns": ret,
      "qp_obs": batch["qp_obs"].astype(jnp.float32), "qp_topk_ids": batch["qp_topk_ids"].astype(jnp.int32)}
grads = jax.grad(lambda tp: brax_ppo_loss(tp, params, hp, ub, 0.2, 0.01, 0.5, 0.0, True, model.get_action_and_value)[0])(
    _trainable_params(params))
ok("gradients finite with LiDAR geometry", all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(grads)))
upd = jax.jit(make_ppo_update_fn(hp, num_epochs=1, num_minibatches=4, max_grad_norm=0.5,
                                 get_action_and_value_fn=model.get_action_and_value))
new_params, _, _, _ = upd(params, init_ppo_opt_state(params), ub, jax.random.PRNGKey(3), 1e-4, 0.2, 0.01, 0.5, 0.5, 0.0)
ok("PPO update runs, params finite", all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(new_params)))
kw = dict(episodes=1, seeds=(100, 200), policy_action_mean_fn=model.policy_action_mean,
          policy_action_to_env_action_fn=model.policy_action_to_env_action, render_cfg=rc,
          image_history_len=hp.image_history_len)
m_l = evaluate_jax_env(params, hp, env_cfg, geometry=geo, **kw)
m_g = evaluate_jax_env(params, hp, env_cfg, **kw)
ok("eval runs with LiDAR geometry and reports perception", "perception_pos_err" in m_l and np.isfinite(m_l["perception_pos_err"]),
   f"success {m_l['success_rate']:.2f} collision {m_l['collision_rate']:.2f} | pos err {m_l['perception_pos_err']:.3f} m")
ok("ground-truth eval has no perception keys", "perception_pos_err" not in m_g)

print("\nALL PASS" if not failed else f"\nFAILED: {failed}")
sys.exit(1 if failed else 0)