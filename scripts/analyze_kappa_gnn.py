"""How does kappa depend on the graph inputs? Analysis for a trained diff_cbf_gnn checkpoint. Run from the repo root:
    python scripts/analyze_kappa_gnn.py --save-dir <run dir> --checkpoint <run dir>/ckpt_XXXXXXXX.pkl

[CBF] New file, the GNN counterpart of scripts/analyze_kappa.py. Sections 1, 2 and 5 match it, so the two reports
can be compared directly. Collects observations from the policy itself (exploration noise on), then reports:
  1. kappa per QP row (row j = j-th nearest human), and across-frame vs within-frame spread
  2. kappa vs the human it belongs to: distance, closing speed, active humans
  3. counterfactual inputs: remove humans / goal / velocities / history and see how kappa moves
  4. sensitivity: which inputs each kappa responds to, including its own human vs the other humans
  5. does the variation matter: safe action with learned kappa vs kappa frozen at its row mean
Saves to <run dir>/kappa_analysis_gnn/: report.txt, extremes.png (lowest and highest kappa scenes).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax

jax.config.update("jax_enable_x64", True)  # same as trainer/jax_platform.py
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("add", lambda *v: sum(v), replace=True)
OmegaConf.register_new_resolver("mul", lambda *v: math.prod(v), replace=True)

from env.jax_social_nav import config_from_omegaconf, reset_env
from model import get_jax_model_module, jax_cbf_qp
from model.jax_ppo_base import activation, dense
from trainer.jax_checkpoint import load_jax_checkpoint
from trainer.jax_ppo_rollout import init_obs_history, make_rollout_collector

ap = argparse.ArgumentParser()
ap.add_argument("--save-dir", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--envs", type=int, default=64)
ap.add_argument("--steps", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=1024, help="frames per network call")
ap.add_argument("--qp-frames", type=int, default=2048, help="frames used for the frozen-kappa QP comparison")
args = ap.parse_args()

cfg = OmegaConf.load(os.path.join(args.save_dir, "config.yaml"))
if str(cfg.model.type) != "diff_cbf_gnn" or not bool(cfg.model.actor.learn_kappa):
    raise SystemExit("this script needs a diff_cbf_gnn run with learn_kappa: true")
model = get_jax_model_module(cfg.model)
hp = model.hparams_from_config(cfg.model, cfg.model.critic, act_dim=int(cfg.env.act_dim))
env_cfg = config_from_omegaconf(cfg)
payload = load_jax_checkpoint(args.checkpoint)
params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
out_dir = os.path.join(args.save_dir, "kappa_analysis_gnn")
os.makedirs(out_dir, exist_ok=True)
lines = []


def say(text=""):
    print(text)
    lines.append(text)


say(f"checkpoint step {payload['step']}, kappa bounds [{hp.kappa_min}, {hp.kappa_max}], init {hp.kappa_init}")

# ---- collect frames
state = reset_env(env_cfg, jax.random.PRNGKey(args.seed), args.envs)
obs_hist = init_obs_history(env_cfg, state, hp.history_len)
collect = make_rollout_collector(model, use_qp=True)
env_state = (state, obs_hist, jnp.zeros((args.envs,), jnp.float32))
batch, _, _ = jax.jit(lambda p, es, k: collect(p, hp, env_cfg, es, k, args.steps))(
    params, env_state, jax.random.PRNGKey(args.seed + 1))
obs = np.asarray(batch["obs"], np.float32)
qp_obs = np.asarray(batch["qp_obs"], np.float32)
topk_ids = np.asarray(batch["qp_topk_ids"], np.int32)
N, K, T, H, F = obs.shape[0], hp.qp_obs_top_k, hp.history_len, hp.num_humans, hp.obs_frame_dim


def kappa_topk(p, ob, ids):
    """The learned kappas for fixed QP rows. ids are held fixed so a counterfactual cannot reorder the rows."""
    _latent, human_in = model._encode_with_humans(p, hp, ob)
    x = activation(dense(p["actor"]["human_fc1"], human_in), hp.actor_act)
    logits = dense(p["actor"]["human_kappa_head"], x)[..., 0]
    return model._kappa_from_logits(model._gather_topk(logits, ids), hp)


kappa_fn = jax.jit(lambda p, o, i: kappa_topk(p, o, i))


def kappas(ob, ids=None):
    ids = topk_ids if ids is None else ids
    return np.concatenate([np.asarray(kappa_fn(params, ob[i:i + args.chunk], ids[i:i + args.chunk]))
                           for i in range(0, len(ob), args.chunk)])


kap = kappas(obs)
blocks = qp_obs[:, 6:].reshape(N, K, 6)
active = blocks[:, :, 5] > 0.5
rel = blocks[:, :, :2]                                   # robot - human, world frame
dist = np.linalg.norm(rel, axis=2)
unit = rel / np.maximum(dist[:, :, None], 1e-6)
closing = -np.sum(unit * (qp_obs[:, None, 2:4] - blocks[:, :, 2:4]), axis=2)    # > 0: gap shrinking
frames = obs.reshape(N, T, F)
n_active = (frames[:, -1, 6:].reshape(N, H, 6)[:, :, 5] > 0.5).sum(axis=1)
masked = np.where(active, kap, np.nan)

# ---- 1. per row
say(f"\n1. kappa per QP row over {N} frames")
say("   row (1 = nearest)   mean    std     min     max    active frac")
for j in range(K):
    v = kap[active[:, j], j]
    if len(v):
        say(f"   {j + 1:<18d} {v.mean():.3f}  {v.std():.3f}  {v.min():.3f}  {v.max():.3f}   {active[:, j].mean():.2f}")
say(f"   std across frames of each frame's mean kappa (effect shared by all rows): {np.nanstd(np.nanmean(masked, axis=1)):.3f}")
say(f"   mean std across rows within one frame (differences between humans):       {np.nanmean(np.nanstd(masked, axis=1)):.3f}")
say(f"   fraction of active kappas within 5% of kappa_max: {np.mean(kap[active] > hp.kappa_max - 0.05 * (hp.kappa_max - hp.kappa_min)):.2f}")

# ---- 2. scene dependence
say("\n2. kappa vs the scene (Spearman rank correlation, active rows)")


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


say("   row   vs distance   vs closing speed   vs active humans")
for j in range(K):
    m = active[:, j]
    say(f"   {j + 1:<5d} {spearman(kap[m, j], dist[m, j]):+.2f}          {spearman(kap[m, j], closing[m, j]):+.2f}"
        f"               {spearman(kap[m, j], n_active[m]):+.2f}")
say("\n   all active rows by distance to that human, split by approaching (closing speed > 0) or receding")
say("   distance        approaching: n, mean        receding: n, mean")
edges = [0, 1, 1.5, 2, 3, 4, 6, np.inf]
for lo, hi in zip(edges[:-1], edges[1:]):
    d = active & (dist >= lo) & (dist < hi)
    a, r = d & (closing > 0), d & (closing <= 0)
    fmt = lambda m: f"{m.sum():7d}, {kap[m].mean():.3f}" if m.sum() else f"{0:7d},   -  "
    say(f"   {lo:4.1f} - {hi:4.1f} m   {fmt(a)}             {fmt(r)}")

# ---- 3. counterfactual inputs
say("\n3. counterfactual inputs, QP rows held fixed (mean over active rows; compare |change| with the natural std above)")
HUMAN_START = 6


def human_view(o):
    return o.reshape(N, T, F)[:, :, HUMAN_START:].reshape(N, T, H, 6)


def rebuild(o, humans):
    fr = o.reshape(N, T, F).copy()
    fr[:, :, HUMAN_START:] = humans.reshape(N, T, H * 6)
    return fr.reshape(N, -1)


rows = np.arange(N)[:, None]
keep_topk = np.zeros((N, H), bool)
keep_topk[rows, topk_ids] = True
h0 = human_view(obs)

v = h0.copy()
v[:, :, :, 5] = v[:, :, :, 5] * keep_topk[:, None, :]
far_removed = rebuild(obs, v)

alone = np.zeros_like(kap)                               # each QP row with every other human removed
for j in range(K):
    keep = np.zeros((N, H), bool)
    keep[np.arange(N), topk_ids[:, j]] = True
    v = h0.copy()
    v[:, :, :, 5] = v[:, :, :, 5] * keep[:, None, :]
    alone[:, j] = kappas(rebuild(obs, v))[:, j]

v = h0.copy()
v[:, :, :, 2:4] = 0.0
no_human_vel = rebuild(obs, v)

fr = frames.copy()
fr[:, :, 0:2] = 0.0
no_goal = fr.reshape(N, -1)

fr = frames.copy()
fr[:, :, 2:4] = 0.0
no_robot_vel = fr.reshape(N, -1)

no_history = np.repeat(frames[:, -1:], T, axis=1).reshape(N, -1)

perm = np.random.default_rng(args.seed).permutation(H)
inverse = np.argsort(perm)
shuffled = rebuild(obs, h0[:, :, perm])

say("   input change                       mean kappa    mean |change|")
say(f"   {'unchanged':<34s} {kap[active].mean():.3f}         -")
for name, k2 in (
    ("humans outside top-k removed", kappas(far_removed)),
    ("each row alone (others removed)", alone),
    ("human velocities zeroed", kappas(no_human_vel)),
    ("goal at the robot", kappas(no_goal)),
    ("robot velocity zeroed", kappas(no_robot_vel)),
    ("no history (newest frame x3)", kappas(no_history)),
    ("slots shuffled (should be ~0)", kappas(shuffled, inverse[topk_ids])),
):
    say(f"   {name:<34s} {k2[active].mean():.3f}         {np.abs(k2 - kap)[active].mean():.3f}")

# ---- 4. sensitivity
sal_ids = np.random.default_rng(args.seed + 3).choice(N, size=min(256, N), replace=False)
ob_s, ids_s = jnp.asarray(obs[sal_ids]), jnp.asarray(topk_ids[sal_ids])
grad_fn = jax.jit(lambda j, o, i: jax.grad(lambda x: jnp.sum(kappa_topk(params, x, i)[:, j]))(o), static_argnums=0)
# |gradient| x natural std of that input, so metres, m/s and radians are comparable
scale = obs.std(axis=0).reshape(T, F)
groups = {"own position": 0.0, "own velocity": 0.0, "other humans": 0.0, "goal": 0.0, "robot velocity": 0.0, "heading": 0.0}
frame_share = np.zeros(T)
S = len(sal_ids)
for j in range(K):
    g = np.abs(np.asarray(grad_fn(j, ob_s, ids_s))).reshape(S, T, F) * scale
    g = g * active[sal_ids, j][:, None, None]
    groups["goal"] += g[:, :, 0:2].sum()
    groups["robot velocity"] += g[:, :, 2:4].sum()
    groups["heading"] += g[:, :, 4].sum()
    hg = g[:, :, 6:].reshape(S, T, H, 6)
    hg[:, :, :, 4:] = 0.0                                # radius and mask are not motion or geometry signals here
    own = np.zeros((S, H), bool)
    own[np.arange(S), topk_ids[sal_ids, j]] = True
    groups["own position"] += hg[:, :, :, 0:2][own[:, None, :].repeat(T, axis=1)].sum()
    groups["own velocity"] += hg[:, :, :, 2:4][own[:, None, :].repeat(T, axis=1)].sum()
    groups["other humans"] += hg[~own[:, None, :].repeat(T, axis=1)].sum()
    frame_share += g[:, :, 0:4].sum(axis=(0, 2)) + hg.sum(axis=(0, 2, 3))
total = max(sum(groups.values()), 1e-12)
say("\n4. sensitivity share, |d kappa / d input| x natural std of that input")
say("   " + "  ".join(f"{name} {val / total:.2f}" for name, val in groups.items()))
frame_share = frame_share / max(frame_share.sum(), 1e-12)
say("   frame:  " + "  ".join(f"t-{T - 1 - i} {frame_share[i]:.2f}" for i in range(T)))

# ---- 5. does the variation change the action?
ids = np.random.default_rng(args.seed + 2).choice(N, size=min(args.qp_frames, N), replace=False)
row_mean = np.array([kap[active[:, j], j].mean() if active[:, j].any() else hp.kappa_init for j in range(K)], np.float32)


@jax.jit
def compare(ob, ctx, frozen):
    _l, qo, nominal, k, rs, safe, slack = model._actor_outputs(params, hp, ob, qp_context=ctx)
    safe2, slack2 = jax_cbf_qp.solve_slack_qp(params, hp, qo, nominal, jnp.broadcast_to(frozen, k.shape), rs)
    return safe, safe2, slack, slack2, nominal, k


ctx = {"qp_obs": jnp.asarray(qp_obs[ids]), "qp_topk_ids": jnp.asarray(topk_ids[ids])}
safe, safe2, sl, sl2, nominal, k_check = compare(jnp.asarray(obs[ids]), ctx, jnp.asarray(row_mean))
assert np.allclose(np.asarray(k_check), kap[ids], atol=1e-4), "kappa path differs from the model"
du = np.linalg.norm(np.asarray(safe) - np.asarray(safe2), axis=1)
cor = np.linalg.norm(np.asarray(safe) - np.asarray(nominal), axis=1)
say(f"\n5. safe action, learned kappa vs kappa frozen at its row mean ({len(ids)} frames)")
say(f"   mean |du| {du.mean():.4f}   95th pct {np.percentile(du, 95):.4f}   max {du.max():.4f}")
say(f"   for scale: mean QP correction |u_safe - u_nominal| = {cor.mean():.4f}")
say(f"   mean slack: learned {float(jnp.mean(sl)):.4f}, frozen {float(jnp.mean(sl2)):.4f}")

# ---- figure: scenes in the robot body frame (robot at the origin, heading along +x)
order = np.argsort(np.nanmean(masked, axis=1))
n_show = 6


def body(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]], axis=-1)


fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5.6))
for row, (shown, title) in enumerate(((order[:n_show], "lowest"), (order[-n_show:], "highest"))):
    for ax, i in zip(axes[row], shown):
        theta = frames[i, -1, 4]
        hb = frames[i, -1, 6:].reshape(H, 6)
        on = hb[:, 5] > 0.5
        p = body(-hb[on, :2], theta)                     # rel = robot - human, so the human is at -rel
        ax.scatter(p[:, 0], p[:, 1], s=14, c="0.75")
        q = body(-rel[i][active[i]], theta)
        ax.scatter(q[:, 0], q[:, 1], s=24, c="tab:red")
        for (x, y), val in zip(q, kap[i][active[i]]):
            ax.annotate(f"{val:.2f}", (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points", clip_on=True)
        g = body(-frames[i, -1, 0:2], theta)             # frame stores robot - goal
        ax.scatter([g[0]], [g[1]], marker="*", s=60, c="tab:green")
        ax.scatter([0], [0], marker=">", s=40, c="tab:blue")
        ax.set_xlim(-6, 6), ax.set_ylim(-6, 6), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{title} {np.nanmean(masked[i]):.2f}\nnearest {dist[i, 0]:.1f} m", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "extremes.png"), dpi=150)
plt.close(fig)

with open(os.path.join(out_dir, "report.txt"), "w") as handle:
    handle.write("\n".join(lines) + "\n")
print(f"saved to {out_dir}")
