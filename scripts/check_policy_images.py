"""Check and save the clean policy images used by ppo_cnn.

[IMG] New file: standalone checks for env/jax_image_renderer.py.

Run from the repo root:
    python scripts/check_policy_images.py                     # 20 humans, saves to outputs/policy_images
    python scripts/check_policy_images.py --view world --size 96 --extent 8

Checks (all on real env states):
  1. shape (B,H,W,3), dtype uint8, values in [0,255], jit works
  2. robot is at the image centre, heading tick points up (egocentric)
  3. inactive human slots draw nothing; image ignores every state field it should not see
  4. egocentric image is invariant to rotating/translating the whole world
  5. frame history is ordered oldest -> newest and fully cleared on reset
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("add", lambda *v: sum(v), replace=True)
OmegaConf.register_new_resolver("mul", lambda *v: math.prod(v), replace=True)

from env.jax_image_renderer import (
    RenderConfig, history_to_cnn_input, init_image_history, meters_per_pixel,
    render_policy_frame, update_image_history,
)
from env.jax_social_nav import config_from_omegaconf, reset_env, step_env

COLORS = np.array([[0.20, 0.45, 1.00],   # robot  -> blue
                   [0.10, 0.80, 0.20],   # goal   -> green
                   [1.00, 0.25, 0.20]])  # humans -> red


def to_rgb(img_u8):
    x = img_u8.astype(np.float32) / 255.0                      # (H,W,3) channels
    return np.clip(1.0 - np.einsum("hwc,ck->hwk", x, 1.0 - COLORS), 0, 1)   # white background


def ok(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        ok.failed = True
ok.failed = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", type=int, default=20)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--extent", type=float, default=7.0)
    ap.add_argument("--view", default="egocentric", choices=["egocentric", "world"])
    ap.add_argument("--out", default="outputs/policy_images")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
        cfg = compose(config_name="config", overrides=["model=ppo_mlp", f"env.humans.num_humans={args.humans}"])
    env_cfg = config_from_omegaconf(cfg)
    rc = RenderConfig(image_size=args.size, half_extent=args.extent, view=args.view)
    render = jax.jit(lambda s: render_policy_frame(env_cfg, s, rc))
    step = jax.jit(lambda s, a, k: step_env(env_cfg, s, a, k))
    reset = jax.jit(lambda k: reset_env(env_cfg, k, args.envs))
    res = meters_per_pixel(rc)
    S = args.size

    key = jax.random.PRNGKey(0)
    state = reset_env(env_cfg, jax.random.PRNGKey(1), args.envs)

    # ---- 1. basic contract
    img = np.asarray(render(state))
    ok("shape/dtype", img.shape == (args.envs, S, S, 3) and img.dtype == np.uint8, f"{img.shape} {img.dtype}")
    ok("all three channels non-empty", all(img[..., c].max() > 0 for c in range(3)))
    print(f"       {S}x{S} px, {res:.3f} m/px, view={rc.view}")

    # ---- 2. robot at centre, heading up (egocentric)
    if rc.view == "egocentric":
        rob = img[0, ..., 0].astype(float)
        ys, xs = np.nonzero(rob > 127)
        ok("robot centred", abs(xs.mean() + 0.5 - S / 2) < 1 and abs(ys.mean() + 0.5 - S / 2) < 2,
           f"centroid=({xs.mean() + 0.5:.1f},{ys.mean() + 0.5:.1f})")
        # Tick = robot-channel pixels above the top of the robot disc. Require a clearly visible (>=50%) pixel
        # there, and nothing below the disc. Not tied to an exact 127/128 cut-off, which float rounding can flip.
        disc_top_row = int(np.floor(S / 2 - env_cfg.robot_radius / res))
        above = rob[:disc_top_row]
        below = rob[int(np.ceil(S / 2 + env_cfg.robot_radius / res)) + 1:]
        ok("heading tick points up", above.max() >= 128 and below.max() == 0,
           f"max above disc={above.max():.0f}, max below disc={below.max():.0f}")

    # ---- 3a. inactive slots draw nothing (put masked humans right next to the robot)
    m = state.human_mask.at[:, : args.humans // 2].set(0.0)
    near = state.robot_state[:, None, 0:2] + jnp.asarray([1.0, 0.0], jnp.float32)
    pos = state.human_positions.at[:, : args.humans // 2].set(near)
    masked = state._replace(human_mask=m, human_positions=pos)
    dropped = state._replace(human_mask=m, human_positions=pos.at[:, : args.humans // 2].set(1e4))
    ok("masked humans invisible", np.array_equal(np.asarray(render(masked)), np.asarray(render(dropped))))

    # ---- 3b. image ignores fields it must not see (velocities, human goals, returns, step count, ...)
    k = jax.random.split(jax.random.PRNGKey(7), 8)
    noisy = state._replace(
        robot_vel=jax.random.normal(k[0], state.robot_vel.shape),
        robot_action=jax.random.normal(k[1], state.robot_action.shape),
        human_vels=jax.random.normal(k[2], state.human_vels.shape) * 3,
        human_goals=jax.random.normal(k[3], state.human_goals.shape) * 9,
        human_vmaxs=jax.random.uniform(k[4], state.human_vmaxs.shape),
        step_count=state.step_count + 123,
        prev_dist_to_goal=state.prev_dist_to_goal * 0 + 99.0,
        episode_returns=state.episode_returns + 5.0,
    )
    ok("no hidden-state leakage", np.array_equal(img, np.asarray(render(noisy))))

    # ---- 3c. human pixel mass matches the disc areas of in-window active humans
    from env.jax_image_renderer import _to_view
    center = 0.5 * (jnp.asarray(env_cfg.workspace_low) + jnp.asarray(env_cfg.workspace_high))
    hv = np.asarray(jax.vmap(lambda p, r, th: _to_view(p, r, th, center, rc))(
        state.human_positions, state.robot_state[:, 0:2], state.robot_state[:, 2]))
    edge = np.abs(hv).max(-1)
    act = np.asarray(state.human_mask) > 0.5
    r_h = env_cfg.human_radius
    full = ((edge < args.extent - r_h) & act).sum(1)
    part = ((edge < args.extent + r_h) & act).sum(1)
    disc = math.pi * (r_h / res) ** 2
    got = img[..., 2].astype(float).sum((1, 2)) / 255.0
    ok("human pixel area ~ pi r^2 per visible human",
       np.all(got <= part * disc * 1.05 + 1) and np.all(got >= full * disc * 0.6 - 1),   # low bound allows overlap
       f"got={got[:3].round(1)} expect~={(full[:3] * disc).round(1)}")

    # ---- 4. egocentric invariance to a rigid transform of the world
    if rc.view == "egocentric":
        a, t = 0.7, jnp.asarray([1.3, -2.1], jnp.float32)
        ca, sa = math.cos(a), math.sin(a)
        # Elementwise on purpose: `p @ R.T` runs in TF32 on Ampere+ GPUs by default (~1e-3 relative
        # error -> ~0.06 px at 10 m), which made this check fail on GPU while the renderer was fine.
        tf = lambda p: jnp.stack([ca * p[..., 0] - sa * p[..., 1],
                                  sa * p[..., 0] + ca * p[..., 1]], axis=-1) + t
        moved = state._replace(
            robot_state=state.robot_state.at[:, 0:2].set(tf(state.robot_state[:, 0:2]))
                                          .at[:, 2].add(a),
            goal_pos=tf(state.goal_pos), human_positions=tf(state.human_positions))
        diff = np.abs(img.astype(int) - np.asarray(render(moved)).astype(int)).max()
        ok("rigid-transform invariance", diff <= 2, f"max diff={diff}")

    # ---- 5. history order and reset, on a real rollout with auto-reset
    T = args.history
    hist = init_image_history(render(state), T)
    ok("init history = T copies", all(np.array_equal(hist[:, 0], hist[:, t]) for t in range(T)))
    frames, resets, saved_reset = [np.asarray(hist[:, -1])], 0, None
    last_reset, reset_ok = np.full(args.envs, -10**6), True
    action = jnp.tile(jnp.asarray([[1.2, 0.0]], jnp.float32), (args.envs, 1))   # drive straight -> episodes end
    for i in range(args.steps):
        key, k1, k2 = jax.random.split(key, 3)
        nxt, _, done, _ = step(state, action, k1)
        done_b = np.asarray(done) > 0.5
        fresh = reset(k2)
        nxt = jax.tree_util.tree_map(
            lambda r, n: jnp.where(done_b.reshape((-1,) + (1,) * (n.ndim - 1)), r, n), fresh, nxt)
        frame = render(nxt)
        hist = update_image_history(hist, frame, jnp.asarray(done_b))
        frames.append(np.asarray(frame))
        h = np.asarray(hist)
        for b in range(args.envs):
            if done_b[b]:
                resets += 1
                ok_reset = all(np.array_equal(h[b, t], frames[-1][b]) for t in range(T))
                reset_ok &= ok_reset
                last_reset[b] = i
                if saved_reset is None:
                    saved_reset = (h[b].copy(), np.stack([f[b] for f in frames[-T - 1:-1]]))
        state = nxt
    h = np.asarray(hist)
    clean = [b for b in range(args.envs) if last_reset[b] < args.steps - T]   # no reset in the last T-1 shifts
    order_ok = all(np.array_equal(h[b, t], frames[-T + t][b]) for b in clean for t in range(T))
    ok("history oldest -> newest", len(clean) > 0 and order_ok, f"({len(clean)} envs checked)")
    ok("reset clears history", resets > 0 and reset_ok, f"({resets} resets seen)")

    x = history_to_cnn_input(hist)
    ok("CNN input", x.shape == (args.envs, S, S, 3 * T) and x.dtype == jnp.float32
       and float(x.max()) <= 1.0, f"{x.shape} {x.dtype}")

    # ---- save sample images
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, T, figsize=(2.6 * T, 5.6))
    for t in range(T):
        ax[0, t].imshow(to_rgb(h[1, t]), interpolation="nearest")
        ax[0, t].set_title(f"env 1, t-{T - 1 - t}" if t < T - 1 else "env 1, latest", fontsize=9)
        if saved_reset is not None:
            ax[1, t].imshow(to_rgb(saved_reset[0][t]), interpolation="nearest")
            ax[1, t].set_title(f"after reset, slot {t}", fontsize=9)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Policy frames ({rc.view}, {S}x{S}, {res:.3f} m/px). Blue robot, green goal, red humans",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"frame_stack_{rc.view}.png"), dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(8, 2.9))
    for c, name in enumerate(["ch0 robot", "ch1 goal", "ch2 humans"]):
        ax[c].imshow(h[1, -1, ..., c], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax[c].set_title(name, fontsize=9); ax[c].set_xticks([]); ax[c].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"channels_{rc.view}.png"), dpi=130)
    for b in range(min(4, args.envs)):
        plt.imsave(os.path.join(args.out, f"raw_env{b}_{rc.view}.png"), to_rgb(h[b, -1]))
    print(f"saved images to {args.out}")
    sys.exit(1 if ok.failed else 0)


if __name__ == "__main__":
    main()