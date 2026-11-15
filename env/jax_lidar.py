"""Simulated 2D LiDAR and obstacle detection for the CBF-QP (Task 003 step 3).  # [LIDAR]

Pipeline (all pure JAX, batched over envs, jit/vmap friendly, float32):

    simulate_scan     ray-circle hits against active humans -> ranges (B, N_beams)
    scan_to_points    ranges + robot pose -> world-frame hit points (B, N_beams, 2)
    segment_scan      split consecutive hits at range jumps -> cluster label per beam
    fit_clusters      known-radius circle fit per cluster -> detected centres (B, C, 2)

`detect_obstacles` runs all four. Tracking and velocity estimation live in env/jax_tracker.py.

Ground-truth use (honest list for the PR):
- The scan is generated from the true human positions/radii/mask, which is what a simulated sensor is.
- Velocities are never read here. They can only come from the tracker.
- The robot pose used to place points in the world frame is the true pose (perfect odometry).
- The circle fit assumes the known human radius (cfg.human_radius).

Reference designs: tysik/obstacle_detector (ROS: 2D scan -> segments -> circles -> Kalman tracking) and
abewley/sort (Kalman filter + assignment). This file follows the same steps in a fixed-shape JAX form.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

F32 = jnp.float32


class LidarConfig(NamedTuple):
    num_beams: int = 360
    fov: float = 2.0 * 3.141592653589793  # full circle; smaller values give a forward-facing sensor
    max_range: float = 12.0
    range_noise_std: float = 0.0  # metres, Gaussian, added to each valid range
    dropout_prob: float = 0.0  # chance a valid return is lost
    # segmentation: neighbours join when |p_i - p_{i-1}| < gap_base + gap_scale * range * beam_spacing
    gap_base: float = 0.15
    gap_scale: float = 3.0
    max_clusters: int = 32  # fixed number of cluster slots (clusters beyond this are dropped)
    min_points: int = 1  # clusters with fewer hits are ignored
    # circle fit
    known_radius: float = 0.4
    fit_iters: int = 6
    fit_damping: float = 1e-3


class Detections(NamedTuple):
    centers: jnp.ndarray  # (B, C, 2) world frame
    valid: jnp.ndarray  # (B, C) bool
    counts: jnp.ndarray  # (B, C) hits per cluster
    means: jnp.ndarray  # (B, C, 2) cluster centroid (diagnostic only; biased toward the sensor)


class Scan(NamedTuple):
    ranges: jnp.ndarray  # (B, N) metres, max_range where there is no return
    valid: jnp.ndarray  # (B, N) bool
    points: jnp.ndarray  # (B, N, 2) world frame (robot position where invalid)
    hit_ids: jnp.ndarray  # (B, N) index of the human hit, -1 for none (diagnostic only)


def lidar_config_from_omegaconf(lidar_cfg, env_cfg=None) -> LidarConfig:
    """Build a LidarConfig from a (possibly partial) dict/OmegaConf node; missing keys keep the defaults."""
    base = LidarConfig()
    values = {}
    for name in LidarConfig._fields:
        value = None
        if lidar_cfg is not None:
            value = lidar_cfg.get(name, None) if hasattr(lidar_cfg, "get") else getattr(lidar_cfg, name, None)
        values[name] = getattr(base, name) if value is None else type(getattr(base, name))(value)
    if env_cfg is not None and (lidar_cfg is None or lidar_cfg.get("known_radius", None) is None):
        values["known_radius"] = float(env_cfg.human_radius)
    return LidarConfig(**values)


def beam_angles(cfg: LidarConfig):
    """Beam angles relative to the robot heading, (N,). Full circle: -pi .. pi (exclusive); else centred."""
    n = cfg.num_beams
    full = cfg.fov >= 2.0 * jnp.pi - 1e-6
    step = cfg.fov / n if full else cfg.fov / max(n - 1, 1)
    start = -jnp.pi if full else -0.5 * cfg.fov
    return (start + step * jnp.arange(n, dtype=F32)).astype(F32)


def _is_full_circle(cfg: LidarConfig) -> bool:
    return cfg.fov >= 2.0 * 3.141592653589793 - 1e-6


def simulate_scan(cfg: LidarConfig, robot_xy, robot_theta, human_pos, human_radii, human_mask, key=None) -> Scan:
    """Ray-circle intersection against every active human; each beam keeps its nearest hit.

    robot_xy (B, 2), robot_theta (B,), human_pos (B, H, 2), human_radii (B, H), human_mask (B, H).
    Occlusion is automatic because a beam stops at its first hit. key is needed only when noise or dropout > 0.
    """
    robot_xy = robot_xy.astype(F32)
    world_ang = robot_theta.astype(F32)[:, None] + beam_angles(cfg)[None, :]  # (B, N)
    d = jnp.stack([jnp.cos(world_ang), jnp.sin(world_ang)], axis=-1)  # (B, N, 2)
    oc = (human_pos.astype(F32) - robot_xy[:, None, :])  # (B, H, 2) robot -> human centre
    # Elementwise products, not einsum/matmul: on GPU, JAX runs float32 matmuls in TF32 (10-bit mantissa) by
    # default, which put ~1 beam in 6 on the wrong side of a circle edge. Elementwise math stays full float32.
    dx, dy = d[:, :, None, 0], d[:, :, None, 1]  # (B, N, 1)
    ox, oy = oc[:, None, :, 0], oc[:, None, :, 1]  # (B, 1, H)
    tca = dx * ox + dy * oy  # (B, N, H) projection of the centre on each ray
    perp2 = (dx * oy - dy * ox) ** 2  # squared distance from the centre to the ray (2D cross product)
    disc = human_radii.astype(F32)[:, None, :] ** 2 - perp2
    t_hit = tca - jnp.sqrt(jnp.maximum(disc, 0.0))
    hit = (disc >= 0.0) & (t_hit > 0.0) & (human_mask[:, None, :] > 0.5)
    t_hit = jnp.where(hit, t_hit, jnp.inf)
    ranges = jnp.min(t_hit, axis=-1)
    hit_ids = jnp.argmin(t_hit, axis=-1).astype(jnp.int32)
    valid = ranges < cfg.max_range
    if cfg.range_noise_std > 0.0 or cfg.dropout_prob > 0.0:
        if key is None:
            raise ValueError("simulate_scan needs a key when range_noise_std or dropout_prob > 0")
        k_noise, k_drop = jax.random.split(key)
        if cfg.range_noise_std > 0.0:
            ranges = ranges + cfg.range_noise_std * jax.random.normal(k_noise, ranges.shape, dtype=F32)
        if cfg.dropout_prob > 0.0:
            valid = valid & (jax.random.uniform(k_drop, ranges.shape, dtype=F32) >= cfg.dropout_prob)
        valid = valid & (ranges > 0.0)
    ranges = jnp.where(valid, ranges, cfg.max_range).astype(F32)
    hit_ids = jnp.where(valid, hit_ids, -1)
    points = scan_to_points(cfg, ranges, valid, robot_xy, robot_theta)
    return Scan(ranges=ranges, valid=valid, points=points, hit_ids=hit_ids)


def scan_to_points(cfg: LidarConfig, ranges, valid, robot_xy, robot_theta):
    """Ranges -> world-frame points using the robot pose (this is where ego-motion is removed)."""
    world_ang = robot_theta.astype(F32)[:, None] + beam_angles(cfg)[None, :]
    d = jnp.stack([jnp.cos(world_ang), jnp.sin(world_ang)], axis=-1)
    r = jnp.where(valid, ranges, 0.0)[..., None]
    return (robot_xy.astype(F32)[:, None, :] + r * d).astype(F32)


def segment_scan(cfg: LidarConfig, scan: Scan):
    """Label each beam with a cluster id in [0, max_clusters) or -1.

    A new cluster starts at a valid beam whose previous beam is invalid or too far away. For a full-circle
    sensor the last cluster is merged into the first one when beams N-1 and 0 are neighbours.
    """
    ranges, valid, points = scan.ranges, scan.valid, scan.points
    spacing = cfg.fov / cfg.num_beams
    prev_pts = jnp.roll(points, 1, axis=1)
    prev_valid = jnp.roll(valid, 1, axis=1)
    gap = jnp.linalg.norm(points - prev_pts, axis=-1)
    thr = cfg.gap_base + cfg.gap_scale * ranges * spacing
    joined = valid & prev_valid & (gap < thr)  # joined[i]: beam i continues the cluster of beam i-1
    starts = valid & ~joined.at[:, 0].set(False)  # beam 0 always starts (wrap handled below)
    labels = jnp.cumsum(starts.astype(jnp.int32), axis=1) - 1
    labels = jnp.where(valid, labels, -1)
    if _is_full_circle(cfg):
        last = jnp.max(labels, axis=1, keepdims=True)
        wrap = joined[:, :1] & (last > 0)  # beam 0 joins beam N-1 and they are different clusters
        labels = jnp.where(wrap & (labels == last), 0, labels)
    labels = jnp.where(labels < cfg.max_clusters, labels, -1)
    return labels


def _cluster_sums(values, labels, num_clusters):
    """values (B, N, ...) summed per cluster -> (B, C, ...). Beams with label -1 are dropped."""
    safe = jnp.where(labels >= 0, labels, num_clusters)

    def one(v, lab):
        return jax.ops.segment_sum(v, lab, num_segments=num_clusters + 1)[:num_clusters]

    return jax.vmap(one)(values, safe)


def fit_clusters(cfg: LidarConfig, scan: Scan, labels, robot_xy) -> Detections:
    """Known-radius circle fit per cluster (damped Gauss-Newton on sum_k (|p_k - c| - R)^2).

    The start point is the centroid pushed one radius away from the sensor, so the fit converges to the
    circle behind the visible arc (a short arc also fits a mirrored circle in front of it).
    """
    C, R = cfg.max_clusters, cfg.known_radius
    w = (labels >= 0).astype(F32)
    counts = _cluster_sums(w, labels, C)  # (B, C)
    sums = _cluster_sums(scan.points * w[..., None], labels, C)
    means = sums / jnp.maximum(counts, 1.0)[..., None]
    away = means - robot_xy.astype(F32)[:, None, :]
    away = away / jnp.maximum(jnp.linalg.norm(away, axis=-1, keepdims=True), 1e-6)
    centers = means + R * away
    lab_safe = jnp.where(labels >= 0, labels, 0)

    def gn_step(c, _):
        c_beam = jnp.take_along_axis(c, lab_safe[..., None], axis=1)  # (B, N, 2) centre of each beam's cluster
        diff = scan.points - c_beam
        dist = jnp.maximum(jnp.linalg.norm(diff, axis=-1), 1e-6)
        res = (dist - R) * w
        jac = -(diff / dist[..., None]) * w[..., None]  # d res / d c
        jtj = _cluster_sums(jac[..., :, None] * jac[..., None, :], labels, C)  # (B, C, 2, 2)
        jtr = _cluster_sums(jac * res[..., None], labels, C)  # (B, C, 2)
        jtj = jtj + cfg.fit_damping * (1.0 + counts)[..., None, None] * jnp.eye(2, dtype=F32)
        delta = jnp.linalg.solve(jtj, -jtr[..., None])[..., 0]
        return c + delta, None

    centers, _ = jax.lax.scan(gn_step, centers, None, length=cfg.fit_iters)
    valid = counts >= max(cfg.min_points, 1)
    centers = jnp.where(valid[..., None], centers, 0.0)
    return Detections(centers=centers.astype(F32), valid=valid, counts=counts, means=means.astype(F32))


def detect_obstacles(cfg: LidarConfig, robot_xy, robot_theta, human_pos, human_radii, human_mask, key=None):
    """Full sensor -> detections step. Returns (Detections, Scan, labels)."""
    scan = simulate_scan(cfg, robot_xy, robot_theta, human_pos, human_radii, human_mask, key)
    labels = segment_scan(cfg, scan)
    dets = fit_clusters(cfg, scan, labels, robot_xy)
    return dets, scan, labels


def detect_from_state(cfg: LidarConfig, state, key=None):
    """Convenience wrapper for a JaxSocialNavState (robot_state = [x, y, theta])."""
    return detect_obstacles(
        cfg,
        state.robot_state[:, 0:2],
        state.robot_state[:, 2],
        state.human_positions,
        state.human_radii,
        state.human_mask,
        key,
    )


def noisy_truth_detections(state, max_clusters, noise_std, key):
    """Stage-B ablation: true centres of active humans + Gaussian noise, as if detection were perfect.

    Skips the scan, segmentation and fit, so it isolates the tracker. Slots beyond num_humans stay invalid.
    """
    pos = state.human_positions.astype(F32)
    B, H, _ = pos.shape
    noisy = pos + noise_std * jax.random.normal(key, pos.shape, dtype=F32)
    valid = state.human_mask > 0.5
    pad = max_clusters - H
    if pad < 0:
        raise ValueError("max_clusters must be >= num_humans")
    centers = jnp.concatenate([jnp.where(valid[..., None], noisy, 0.0), jnp.zeros((B, pad, 2), F32)], axis=1)
    valid = jnp.concatenate([valid, jnp.zeros((B, pad), bool)], axis=1)
    counts = valid.astype(F32)
    return Detections(centers=centers, valid=valid, counts=counts, means=centers)