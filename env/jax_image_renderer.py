from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

ROBOT_CH, GOAL_CH, HUMAN_CH = 0, 1, 2
NUM_CHANNELS = 3

class RenderConfig(NamedTuple):
    image_size: int = 84        # H = W, pixels
    half_extent: float = 4.0    # metres from image center to edge 
    view: str = "egocentric"    # "egocentric" = robot centred, heading up | "world" = arena centered, +y up
    goal_radius: float = 0.3     # metres
    goal_clip_to_border: bool = True  # egocentric: goal outside the window is drawn on the border (direction only)
    heading_tick: bool = True  # draw a tick mark on the robot to indicate heading | length = 1.6 * robot radius
    tick_width_px: float = 2.0 

def render_config_from_omegaconf(node) -> RenderConfig:
    """Build from a dict-like config node (e.g. cfg.model.render). Missing keys use defaults."""
    if node is None:
        return RenderConfig()
    get = node.get if hasattr(node, "get") else (lambda k, d=None: getattr(node, k, d))
    base = RenderConfig()
    rc = RenderConfig(**{f: type(getattr(base, f))(get(f, getattr(base, f))) for f in base._fields})
    if rc.view not in {"egocentric", "world"}:
        raise ValueError(f"render.view={rc.view!r} must be 'egocentric' or 'world'")
    return rc

def meters_per_pixel(rc: RenderConfig) -> float:
    return (2.0 * rc.half_extent) / rc.image_size

def _pixel_grid(rc: RenderConfig):
    """(H, W, 2) pixel-centre coordinates in the view frame (u right, v up), meters"""
    s, res = rc.image_size, meters_per_pixel(rc)
    idx = jnp.arange(s, dtype=jnp.float32) + 0.5
    u = (idx - s / 2) * res     # column -> right
    v = (s / 2 - idx) * res     # row -> up
    uu, vv = jnp.meshgrid(u, v, indexing="xy")
    return jnp.stack([uu, vv], axis=-1)  # (H, W, 2)

def _to_view(points, robot_xy, robot_theta, center_xy, rc: RenderConfig):
    """Transform points from world frame to view frame (egocentric or world)"""
    if rc.view == "egocentric":
        # translate to robot frame
        d = points - robot_xy
        # rotate to robot heading
        c, s = jnp.cos(robot_theta), jnp.sin(robot_theta)
        x_b = c * d[..., 0] + s * d[..., 1]
        y_b = -s * d[..., 0] + c * d[..., 1]
        return jnp.stack([-y_b, x_b], axis=-1)  # (N, 2) in view frame (u right, v up)
    
    return points - center_xy

def _soft_disc(grid, centers, radii, res):
    """grid (H, W, 2), centers (K, 2), radii (K,) -> (K, H, W) coverage in [0, 1]"""
    d = jnp.linalg.norm(grid[None] - centers[:, None, None, :], axis=-1)
    return jnp.clip((radii[:, None, None] - d) / res + 0.5, 0.0, 1.0)

def _soft_segment(grid, a, b, width_m, res):
    """Coverage (H, W) of a segment a->b with the given width."""
    ab = b-a
    t = jnp.clip(jnp.sum((grid-a)*ab, -1) / (jnp.sum(ab*ab) + 1e-12), 0.0, 1.0)
    d = jnp.linalg.norm(grid - (a + t[..., None] * ab), axis=-1)
    return jnp.clip((0.5 * width_m - d) / res + 0.5, 0.0, 1.0)

def _render_one(rc: RenderConfig, robot_radius, center_xy, robot_xy, robot_theta, goal_xy, human_xy, human_r, human_mask):
    f32 = jnp.float32
    res = jnp.asarray(meters_per_pixel(rc), dtype=f32)
    half = jnp.asarray(rc.half_extent, dtype=f32)
    grid = _pixel_grid(rc)

    to_view = lambda pts: _to_view(pts, robot_xy, robot_theta, center_xy, rc)
    robot_v = to_view(robot_xy)
    goal_v = to_view(goal_xy)
    human_v = to_view(human_xy)

    # robot channel
    rr = jnp.asarray(robot_radius, dtype=f32)
    robot = _soft_disc(grid, robot_v[None], rr[None], res)[0]
    if rc.heading_tick:
        if rc.view == "egocentric":
            fwd = jnp.asarray([0.0, 1.0], f32)
        else:
            fwd = jnp.stack([jnp.cos(robot_theta), jnp.sin(robot_theta)]).astype(f32)
        tick = _soft_segment(grid, robot_v, robot_v + 1.6 * rr * fwd, rc.tick_width_px * res, res)
        robot = jnp.maximum(robot, tick)

    # goal
    gr = jnp.array(rc.goal_radius, f32)
    if rc.goal_clip_to_border:
        lim = half - gr - res
        scale = jnp.minimum(1.0, lim / jnp.maximum(jnp.max(jnp.abs(goal_v - robot_v)), 1e-6))
        goal_v = robot_v + scale * (goal_v - robot_v)
    goal = _soft_disc(grid, goal_v[None], gr[None], res)[0]

    # humans: anything off window is cropped, mask > 0.5 =  visible
    active = (human_mask > 0.5).astype(f32)
    humans = _soft_disc(grid, human_v, human_r.astype(f32), res) * active[:, None, None]
    humans = jnp.max(humans, axis=0) if humans.shape[0] > 0 else jnp.zeros_like(robot)

    img = jnp.stack([robot, goal, humans], axis=-1)
    return jnp.round(img * 255.0).astype(jnp.uint8)

def render_policy_frame(env_cfg, state, rc: RenderConfig = RenderConfig()):
    """Batched env state -> (B, H, W, 3) uint8 policy image. Uses only the current state"""
    f32 = jnp.float32
    center = (0.5 * (jnp.asarray(env_cfg.workspace_low) + jnp.asarray(env_cfg.workspace_high))).astype(f32)
    fn = lambda rxy, rth, gxy, hxy, hr, hm: _render_one(rc, env_cfg.robot_radius, center, rxy, rth, gxy, hxy, hr, hm)
    return jax.vmap(fn)(
        state.robot_state[:, 0:2].astype(f32),
        state.robot_state[:, 2].astype(f32),
        state.goal_pos.astype(f32),
        state.human_positions.astype(f32),
        state.human_radii.astype(f32),
        state.human_mask.astype(f32),
    )

### Frame History
# Same as trainer.jax_ppo_rollout.init_obs_history / _update_obs_history
# oldest -> newest along axis 1, and on reset every slot is overwritten
# with the first frame of the new epsiode, so nothing from the old episode survives.

def init_image_history(frame, history_len):
    """frame (B, H, W, C) --> (B, T, H, W, C); first frame repeated T times"""
    return jnp.repeat(frame[:, None], int(history_len), axis=1)

def update_image_history(history, next_frame, reset_mask):
    """Shifted in the next frame: envs with reset_mask=True get T copies of next_frame"""
    shifted = jnp.concatenate([history[:, 1:], next_frame[:, None]], axis=1)
    fresh = jnp.repeat(next_frame[:, None], history.shape[1], axis=1)
    return jnp.where(jnp.asarray(reset_mask, bool)[:, None, None, None, None], fresh, shifted)

def history_to_cnn_input(history):
    """(B, T, H, W, C) unit8 -> (B, H, W, T*C) float32 in [0, 1], channels ordered (t0c0, t0c1, t0c2, t1c0, ...)"""
    b, t, h, w, c = history.shape
    x = jnp.transpose(history, (0, 2, 3, 1, 4)).reshape((b, h, w, t*c))
    return x.astype(jnp.float32) / 255.0


