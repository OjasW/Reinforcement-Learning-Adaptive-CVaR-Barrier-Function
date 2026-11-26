"""How does kappa depend on the images? Analysis for a trained diff_cbf_cnn checkpoint. Run from the repo root:
    python scripts/analyze_kappa.py --save-dir <run dir> --checkpoint <run dir>/ckpt_XXXXXXXX.pkl

[CBF] New file. Collects images from the policy itself (exploration noise on), then reports:
  1. kappa per QP row (row j = j-th nearest human), and across-frame vs within-frame spread
  2. kappa vs what the scene looks like: distance, closing speed, humans in the window
  3. counterfactual images: erase the humans / goal / robot, or remove motion, and see how kappa moves
  4. saliency: which image channel and which of the stacked frames kappa is most sensitive to
  5. does the variation matter: safe action with learned kappa vs kappa frozen at its row mean
Saves to <run dir>/kappa_analysis/: report.txt, extremes.png (lowest and highest kappa frames), saliency.png.
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

from env.jax_image_renderer import history_to_cnn_input, init_image_history, render_config_from_omegaconf, render_policy_frame
from env.jax_social_nav import config_from_omegaconf, reset_env
from model import get_jax_model_module, jax_cbf_qp
from model import jax_ppo_cnn as cnn
from model.jax_ppo_base import activation
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
if str(cfg.model.type) != "diff_cbf_cnn" or not bool(cfg.model.actor.learn_kappa):
    raise SystemExit("this script needs a diff_cbf_cnn run with learn_kappa: true")
model = get_jax_model_module(cfg.model)
hp = model.hparams_from_config(cfg.model, cfg.model.critic, act_dim=int(cfg.env.act_dim))
env_cfg = config_from_omegaconf(cfg)
rc = render_config_from_omegaconf(cfg.model.render)
payload = load_jax_checkpoint(args.checkpoint)
params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
out_dir = os.path.join(args.save_dir, "kappa_analysis")
os.makedirs(out_dir, exist_ok=True)
lines = []


def say(text=""):
    print(text)
    lines.append(text)


say(f"checkpoint step {payload['step']}, kappa bounds [{hp.kappa_min}, {hp.kappa_max}], init {hp.kappa_init}")

# ---- collect frames
state = reset_env(env_cfg, jax.random.PRNGKey(args.seed), args.envs)
obs_hist = init_obs_history(env_cfg, state, hp.history_len)
img0 = init_image_history(render_policy_frame(env_cfg, state, rc), hp.image_history_len)
collect = make_rollout_collector(model, use_qp=True, render_cfg=rc)
env_state = (state, obs_hist, img0, jnp.zeros((args.envs,), jnp.float32))
batch, _, _ = jax.jit(lambda p, es, k: collect(p, hp, env_cfg, es, k, args.steps))(
    params, env_state, jax.random.PRNGKey(args.seed + 1))
images = np.asarray(batch["img"])
qp_obs = np.asarray(batch["qp_obs"], np.float32)
obs = np.asarray(batch["obs"], np.float32)
N, K = images.shape[0], hp.qp_obs_top_k
kappa_fn = jax.jit(lambda p, im: model._network_outputs(p, hp, im)[2])


def kappas(im):
    return np.concatenate([np.asarray(kappa_fn(params, im[i:i + args.chunk])) for i in range(0, len(im), args.chunk)])


kap = kappas(images)
blocks = qp_obs[:, 6:].reshape(N, K, 6)
active = blocks[:, :, 5] > 0.5
theta = qp_obs[:, 4]
rel = blocks[:, :, :2]                                   # robot - human, world frame
dist = np.linalg.norm(rel, axis=2)
unit = rel / np.maximum(dist[:, :, None], 1e-6)
closing = -np.sum(unit * (qp_obs[:, None, 2:4] - blocks[:, :, 2:4]), axis=2)    # > 0: gap shrinking
c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
body_x, body_y = c * rel[:, :, 0] + s * rel[:, :, 1], -s * rel[:, :, 0] + c * rel[:, :, 1]
in_window = (np.maximum(np.abs(body_x), np.abs(body_y)) <= rc.half_extent) & active
masked = np.where(active, kap, np.nan)

# ---- 1. per row
say(f"\n1. kappa per QP row over {N} frames")
say("   row (1 = nearest)   mean    std     min     max    active frac   in-window frac")
for j in range(K):
    v = kap[active[:, j], j]
    if len(v):
        say(f"   {j + 1:<18d} {v.mean():.3f}  {v.std():.3f}  {v.min():.3f}  {v.max():.3f}   "
            f"{active[:, j].mean():.2f}          {in_window[:, j].mean():.2f}")
say(f"   std across frames of each frame's mean kappa (image effect shared by all rows): {np.nanstd(np.nanmean(masked, axis=1)):.3f}")
say(f"   mean std across rows within one frame (differences between humans):             {np.nanmean(np.nanstd(masked, axis=1)):.3f}")
say(f"   fraction of active kappas within 5% of kappa_max: {np.mean(kap[active] > hp.kappa_max - 0.05 * (hp.kappa_max - hp.kappa_min)):.2f}")

# ---- 2. scene dependence
say("\n2. kappa vs the scene (Spearman rank correlation, active rows)")


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


n_vis = in_window.sum(axis=1)
say("   row   vs distance   vs closing speed   vs humans in window")
for j in range(K):
    m = active[:, j]
    say(f"   {j + 1:<5d} {spearman(kap[m, j], dist[m, j]):+.2f}          {spearman(kap[m, j], closing[m, j]):+.2f}"
        f"               {spearman(kap[m, j], n_vis[m]):+.2f}")
say("\n   row 1 kappa by distance to the nearest human")
edges = [0, 1, 1.5, 2, 3, 4, rc.half_extent * math.sqrt(2), np.inf]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = active[:, 0] & (dist[:, 0] >= lo) & (dist[:, 0] < hi)
    if m.sum():
        say(f"   {lo:4.1f} - {hi:4.1f} m   n={m.sum():6d}   mean {kap[m, 0].mean():.3f}   std {kap[m, 0].std():.3f}")

# ---- 3. counterfactual images
say("\n3. counterfactual images (mean over active rows; compare |change| with the natural std above)")
ROBOT, GOAL, HUMANS = 0, 1, 2
variants = {}
for name, ch in (("no humans", HUMANS), ("no goal", GOAL), ("no robot", ROBOT)):
    v = images.copy()
    v[..., ch] = 0
    variants[name] = v
variants["no motion (newest frame x4)"] = np.repeat(images[:, -1:], images.shape[1], axis=1)
say("   image change                 mean kappa    mean |change|")
say(f"   {'unchanged':<28s} {kap[active].mean():.3f}         -")
for name, v in variants.items():
    k2 = kappas(v)
    say(f"   {name:<28s} {k2[active].mean():.3f}         {np.abs(k2 - kap)[active].mean():.3f}")

# ---- 4. saliency
order = np.argsort(np.nanmean(masked, axis=1))
sal_ids = np.random.default_rng(args.seed).choice(N, size=min(256, N), replace=False)


def mean_kappa_from_float(x):
    a = params["actor"]
    for layer, stride in zip(a["cnn"]["conv"], hp.cnn_strides):
        x = activation(cnn._conv(layer, x, stride), hp.actor_act)
    x = activation(cnn._lin(a["cnn"]["fc"], x.reshape((x.shape[0], -1))), hp.actor_act)
    x = activation(cnn._lin(a["kappa_fc"], x), hp.actor_act)
    return model._kappa_from_logits(cnn._lin(a["kappa_head"], x), hp)


saliency_fn = jax.jit(jax.grad(lambda x: jnp.sum(mean_kappa_from_float(x))))
x_in = history_to_cnn_input(jnp.asarray(images[sal_ids])).astype(jnp.float32)
assert np.allclose(np.asarray(mean_kappa_from_float(x_in)), kap[sal_ids], atol=1e-4), "saliency path differs from the model"
sal = np.abs(np.asarray(saliency_fn(x_in)))              # (S, H, W, 3T)
T = hp.image_history_len
per = sal.reshape(sal.shape[0], sal.shape[1], sal.shape[2], T, 3)       # frame order follows history_to_cnn_input
tot = per.sum()
say("\n4. saliency share of |d kappa / d pixel|")
say("   channel:  " + "  ".join(f"{n} {per[..., i].sum() / tot:.2f}" for i, n in enumerate(("robot", "goal", "humans"))))
say("   frame:    " + "  ".join(f"t-{T - 1 - i} {per[:, :, :, i].sum() / tot:.2f}" for i in range(T)))

# ---- 5. does the variation change the action?
ids = np.random.default_rng(args.seed + 2).choice(N, size=min(args.qp_frames, N), replace=False)
row_mean = np.array([kap[active[:, j], j].mean() if active[:, j].any() else hp.kappa_init for j in range(K)], np.float32)
ctx_ids = lambda: {"qp_obs": jnp.asarray(qp_obs[ids]), "qp_topk_ids": jnp.asarray(batch["qp_topk_ids"])[ids]}


@jax.jit
def compare(ob, im, ctx, frozen):
    _l, qo, nominal, k, rs, safe, slack = model._actor_outputs(params, hp, ob, im, qp_context=ctx)
    safe2, slack2 = jax_cbf_qp.solve_slack_qp(params, hp, qo, nominal, jnp.broadcast_to(frozen, k.shape), rs)
    return safe, safe2, slack, slack2, nominal


safe, safe2, sl, sl2, nominal = compare(jnp.asarray(obs[ids]), jnp.asarray(images[ids]), ctx_ids(), jnp.asarray(row_mean))
du = np.linalg.norm(np.asarray(safe) - np.asarray(safe2), axis=1)
cor = np.linalg.norm(np.asarray(safe) - np.asarray(nominal), axis=1)
say(f"\n5. safe action, learned kappa vs kappa frozen at its row mean ({len(ids)} frames)")
say(f"   mean |du| {du.mean():.4f}   95th pct {np.percentile(du, 95):.4f}   max {du.max():.4f}")
say(f"   for scale: mean QP correction |u_safe - u_nominal| = {cor.mean():.4f}")
say(f"   mean slack: learned {float(jnp.mean(sl)):.4f}, frozen {float(jnp.mean(sl2)):.4f}")

# ---- figures
COL = np.array([[0.13, 0.35, 0.8], [0.1, 0.65, 0.25], [0.85, 0.15, 0.15]])


def to_rgb(frame):
    a = frame.astype(np.float32)[..., None] / 255.0
    rgb = np.ones(frame.shape[:2] + (3,))
    for ch in range(3):
        rgb = rgb * (1 - a[:, :, ch]) + a[:, :, ch] * COL[ch]
    return rgb


n_show = 6
fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5))
for row, (ids_show, title) in enumerate(((order[:n_show], "lowest"), (order[-n_show:], "highest"))):
    for ax, i in zip(axes[row], ids_show):
        ax.imshow(to_rgb(images[i, -1]))
        ax.set_title(f"{title} {np.nanmean(masked[i]):.2f}\nnearest {dist[i, 0]:.1f} m", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "extremes.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
axes[0].imshow(to_rgb(images[sal_ids[0], -1]))
axes[0].set_title("newest frame (one sample)", fontsize=9)
axes[1].imshow(per.mean(axis=0).sum(axis=(2, 3)), cmap="magma")
axes[1].set_title("mean saliency over sampled frames", fontsize=9)
for ax in axes:
    ax.set_xticks([]), ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "saliency.png"), dpi=150)
plt.close(fig)

with open(os.path.join(out_dir, "report.txt"), "w") as handle:
    handle.write("\n".join(lines) + "\n")
print(f"saved to {out_dir}")
