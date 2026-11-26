"""Multi-object tracker and LiDAR-based QP observation frame (Task 003 step 3).  # [LIDAR]

Constant-velocity Kalman filter per track, in the WORLD frame (robot motion is removed when the scan is
converted to world points), with gated greedy nearest-neighbour association and fixed-size slots so that
everything is jit/vmap friendly.

    tracker = init_tracker(tcfg, batch)
    tracker = tracker_step(tcfg, tracker, detections, reset_mask)   # once per env step
    frame   = lidar_policy_obs(env_cfg, state, tracker, tcfg)       # same layout as fixed_order_policy_obs_from_state

The frame keeps the env's layout (6 robot values + num_humans x [robot - p, v, radius, mask]), so the
diff_cbf_cnn QP code reads it without changes. Only the 20 human blocks are estimates; the 6 robot values
(goal offset, robot velocity, heading, robot radius) are still the true robot state.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from env import jax_lidar as L

F32 = jnp.float32
HIGHEST = jax.lax.Precision.HIGHEST  # full float32 matmuls on GPU (default there is TF32, ~3 significant digits)
LIDAR_KEY_SALT = 1729  # fold_in salt: sensor noise keys never change the env/policy random streams


class TrackerConfig(NamedTuple):
    num_slots: int = 20  # = env num_humans, so the frame has the same size as the ground-truth frame
    dt: float = 0.1
    gate: float = 1.0  # metres; a detection farther than this from a predicted track starts a new track
    max_coast: int = 3  # steps a track survives without a detection (bridges short occlusions)
    min_hits: int = 1  # updates before a track is shown to the QP (1 = immediately)
    accel_std: float = 3.0  # m/s^2, white-acceleration noise; env humans change velocity ~2.3 m/s^2 on average
    meas_std: float = 0.05  # m, detection position noise
    init_vel_std: float = 1.5  # m/s, initial velocity uncertainty (new tracks start at v = 0)
    known_radius: float = 0.4


class TrackerState(NamedTuple):
    x: jnp.ndarray  # (B, S, 4) [px, py, vx, vy] world frame
    P: jnp.ndarray  # (B, S, 4, 4)
    active: jnp.ndarray  # (B, S) bool
    hits: jnp.ndarray  # (B, S) int32, number of updates
    misses: jnp.ndarray  # (B, S) int32, consecutive steps without a detection
    track_ids: jnp.ndarray  # (B, S) int32, unique id per spawned track (diagnostics / ID switches)
    next_id: jnp.ndarray  # (B,) int32


def tracker_config_from_omegaconf(tracker_cfg, env_cfg) -> TrackerConfig:
    base = TrackerConfig()
    values = {}
    for name in TrackerConfig._fields:
        value = None
        if tracker_cfg is not None:
            value = tracker_cfg.get(name, None) if hasattr(tracker_cfg, "get") else getattr(tracker_cfg, name, None)
        values[name] = getattr(base, name) if value is None else type(getattr(base, name))(value)
    values["num_slots"] = int(env_cfg.num_humans)
    values["dt"] = float(env_cfg.dt)
    if tracker_cfg is None or tracker_cfg.get("known_radius", None) is None:
        values["known_radius"] = float(env_cfg.human_radius)
    return TrackerConfig(**values)


def init_tracker(cfg: TrackerConfig, batch_size: int) -> TrackerState:
    B, S = batch_size, cfg.num_slots
    return TrackerState(
        x=jnp.zeros((B, S, 4), F32),
        P=jnp.broadcast_to(jnp.eye(4, dtype=F32), (B, S, 4, 4)),
        active=jnp.zeros((B, S), bool),
        hits=jnp.zeros((B, S), jnp.int32),
        misses=jnp.zeros((B, S), jnp.int32),
        track_ids=jnp.full((B, S), -1, jnp.int32),
        next_id=jnp.zeros((B,), jnp.int32),
    )


def _cv_matrices(cfg: TrackerConfig):
    dt = cfg.dt
    F = jnp.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], F32)
    q = cfg.accel_std ** 2
    q1 = jnp.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]], F32) * q  # per axis
    Q = jnp.zeros((4, 4), F32)
    Q = Q.at[jnp.ix_(jnp.array([0, 2]), jnp.array([0, 2]))].set(q1)
    Q = Q.at[jnp.ix_(jnp.array([1, 3]), jnp.array([1, 3]))].set(q1)
    H = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0]], F32)
    Rm = (cfg.meas_std ** 2) * jnp.eye(2, dtype=F32)
    return F, Q, H, Rm


def _greedy_assign(cost, gate, num_iters):
    """cost (S, C) -> det index per slot (S,), -1 if unassigned. Repeatedly takes the global minimum."""
    S, C = cost.shape

    def body(_, carry):
        cost, assign = carry
        flat = jnp.argmin(cost).astype(jnp.int32)  # argmin is int64 under x64; assign is int32
        s, c = flat // C, flat % C
        ok = cost[s, c] < gate
        assign = jnp.where(ok, assign.at[s].set(c), assign)
        cost = jnp.where(ok, cost.at[s, :].set(jnp.inf).at[:, c].set(jnp.inf), cost)
        return cost, assign

    _, assign = jax.lax.fori_loop(0, num_iters, body, (cost, jnp.full((S,), -1, jnp.int32)))
    return assign


def _step_one(cfg: TrackerConfig, F, Q, H, Rm, x, P, active, hits, misses, ids, next_id, z, z_valid, reset):
    S = x.shape[0]
    C = z.shape[0]
    # 0) episode reset clears every track
    active = active & ~reset
    hits = jnp.where(reset, 0, hits)
    misses = jnp.where(reset, 0, misses)
    ids = jnp.where(reset, -1, ids)

    # 1) predict
    x_pred = jnp.einsum("ij,sj->si", F, x, precision=HIGHEST)
    P_pred = jnp.einsum("ij,sjk,lk->sil", F, P, F, precision=HIGHEST) + Q
    x = jnp.where(active[:, None], x_pred, x)
    P = jnp.where(active[:, None, None], P_pred, P)

    # 2) associate (Euclidean distance between predicted position and detection, gated)
    dist = jnp.linalg.norm(x[:, None, 0:2] - z[None, :, :], axis=-1)
    cost = jnp.where(active[:, None] & z_valid[None, :], dist, jnp.inf)
    assign = _greedy_assign(cost, cfg.gate, min(S, C))
    matched = assign >= 0

    # 3) Kalman update for matched tracks
    z_s = z[jnp.maximum(assign, 0)]  # (S, 2)
    innov = z_s - x[:, 0:2]
    Sm = P[:, 0:2, 0:2] + Rm
    K = jnp.einsum("sij,sjk->sik", P[:, :, 0:2], jnp.linalg.inv(Sm), precision=HIGHEST)  # (S, 4, 2)
    x_upd = x + jnp.einsum("sij,sj->si", K, innov, precision=HIGHEST)
    P_upd = P - jnp.einsum("sij,jk,skl->sil", K, H, P, precision=HIGHEST)
    x = jnp.where(matched[:, None], x_upd, x)
    P = jnp.where(matched[:, None, None], P_upd, P)
    hits = jnp.where(matched, hits + 1, hits)
    misses = jnp.where(matched, 0, jnp.where(active, misses + 1, misses))
    active = active & (misses <= cfg.max_coast)

    # 4) spawn tracks for unmatched detections in free slots (k-th new detection -> k-th free slot)
    det_used = jnp.zeros((C,), bool).at[jnp.where(matched, assign, C)].set(True, mode="drop")
    det_new = z_valid & ~det_used
    free = ~active
    free_slots = jnp.argsort(~free, stable=True)  # free slots first, in index order
    num_free = jnp.sum(free)
    det_rank = (jnp.cumsum(det_new.astype(jnp.int32)) - 1).astype(jnp.int32)  # cumsum gives int64 under x64
    spawn = det_new & (det_rank < num_free)
    slot_for_det = jnp.where(spawn, free_slots[jnp.clip(det_rank, 0, S - 1)], S)  # S = dropped
    x_new = jnp.concatenate([z, jnp.zeros((C, 2), F32)], axis=1)
    P_new = jnp.broadcast_to(
        jnp.diag(jnp.array([cfg.meas_std ** 2] * 2 + [cfg.init_vel_std ** 2] * 2, F32)), (C, 4, 4)
    )
    new_ids = next_id + det_rank
    x = x.at[slot_for_det].set(x_new, mode="drop")
    P = P.at[slot_for_det].set(P_new, mode="drop")
    active = active.at[slot_for_det].set(True, mode="drop")
    hits = hits.at[slot_for_det].set(1, mode="drop")
    misses = misses.at[slot_for_det].set(0, mode="drop")
    ids = ids.at[slot_for_det].set(new_ids, mode="drop")
    next_id = next_id + jnp.sum(spawn).astype(jnp.int32)
    ids = jnp.where(active, ids, -1)
    return x, P, active, hits, misses, ids, next_id


def tracker_step(cfg: TrackerConfig, tracker: TrackerState, detections, reset_mask=None) -> TrackerState:
    """One tracking step. detections: env.jax_lidar.Detections (centers (B, C, 2), valid (B, C)).

    reset_mask (B,) bool clears the tracks of envs that just started a new episode, BEFORE using this
    step's detections (call it with the done mask after the env reset, like the obs/image histories).
    """
    B = tracker.x.shape[0]
    if reset_mask is None:
        reset_mask = jnp.zeros((B,), bool)
    F, Q, H, Rm = _cv_matrices(cfg)
    step = jax.vmap(lambda *a: _step_one(cfg, F, Q, H, Rm, *a))
    x, P, active, hits, misses, ids, next_id = step(
        tracker.x,
        tracker.P,
        tracker.active,
        tracker.hits,
        tracker.misses,
        tracker.track_ids,
        tracker.next_id,
        detections.centers.astype(F32),
        detections.valid,
        reset_mask,
    )
    return TrackerState(x=x, P=P, active=active, hits=hits, misses=misses, track_ids=ids, next_id=next_id)


def visible_mask(cfg: TrackerConfig, tracker: TrackerState):
    """Tracks the QP is allowed to use: active and updated at least min_hits times."""
    return tracker.active & (tracker.hits >= cfg.min_hits)


def lidar_policy_obs(env_cfg, state, tracker: TrackerState, cfg: TrackerConfig):
    """Drop-in replacement for env.jax_social_nav.fixed_order_policy_obs_from_state.

    Robot block (6): true goal offset, robot velocity, heading, robot radius (unchanged).
    Human blocks (num_slots x 6): robot - track position, track velocity (world), known radius, mask.
    Nothing here reads state.human_positions / human_vels / human_mask.
    """
    batch_size = state.robot_state.shape[0]
    robot = jnp.concatenate(
        [
            state.robot_state[:, 0:2] - state.goal_pos,
            state.robot_vel,
            state.robot_state[:, 2:3],
            jnp.full((batch_size, 1), env_cfg.robot_radius, dtype=state.robot_state.dtype),
        ],
        axis=1,
    ).astype(F32)
    mask = visible_mask(cfg, tracker)
    m = mask[..., None].astype(F32)
    rel = (state.robot_state[:, None, 0:2].astype(F32) - tracker.x[..., 0:2]) * m
    vel = tracker.x[..., 2:4] * m
    radius = jnp.full(mask.shape + (1,), cfg.known_radius, F32) * m
    blocks = jnp.concatenate([rel, vel, radius, m], axis=-1)
    return jnp.concatenate([robot, blocks.reshape((batch_size, -1))], axis=1).astype(F32)


def perception_metrics(state, tracker: TrackerState, cfg: TrackerConfig, match_radius=0.5, near_range=None):
    """Per-step perception errors against ground truth (diagnostics only, never fed to the policy/QP).

    Each active human is matched to its nearest visible track (within match_radius). Returns a dict of
    (B,) arrays: pos_err / vel_err (mean over matched humans), missed (active humans with no track),
    false_tracks (visible tracks with no human within match_radius). near_range limits the humans counted
    to those within that distance of the robot (e.g. the ones the QP can select).
    """
    hpos = state.human_positions.astype(F32)
    hvel = state.human_vels.astype(F32)
    hmask = state.human_mask > 0.5
    if near_range is not None:
        rdist = jnp.linalg.norm(hpos - state.robot_state[:, None, 0:2], axis=-1)
        hmask = hmask & (rdist < near_range)
    tmask = visible_mask(cfg, tracker)
    d = jnp.linalg.norm(hpos[:, :, None, :] - tracker.x[:, None, :, 0:2], axis=-1)  # (B, H, S)
    d = jnp.where(tmask[:, None, :], d, jnp.inf)
    nearest = jnp.argmin(d, axis=2)
    dmin = jnp.min(d, axis=2)
    matched = hmask & (dmin < match_radius)
    tvel = jnp.take_along_axis(tracker.x[..., 2:4], nearest[..., None], axis=1)
    verr = jnp.linalg.norm(tvel - hvel, axis=-1)
    n = jnp.maximum(jnp.sum(matched, axis=1), 1)
    pos_err = jnp.sum(jnp.where(matched, dmin, 0.0), axis=1) / n
    vel_err = jnp.sum(jnp.where(matched, verr, 0.0), axis=1) / n
    missed = jnp.sum(hmask & ~matched, axis=1)
    d_t = jnp.where(state.human_mask[:, :, None] > 0.5, d, jnp.inf)
    false_tracks = jnp.sum(tmask & (jnp.min(d_t, axis=1) >= match_radius), axis=1)
    return {
        "pos_err": pos_err,
        "vel_err": vel_err,
        "missed": missed.astype(F32),
        "false_tracks": false_tracks.astype(F32),
        "num_matched": jnp.sum(matched, axis=1).astype(F32),
    }


# ---------------------------------------------------------------- geometry provider used by trainer/eval
class LidarGeometry(NamedTuple):
    """Static settings for estimated QP geometry. source: "lidar" (scan -> fit) or "truth" (stage-B ablation:
    true centres + noise, skipping the scan). perception_range: humans counted in the logged errors."""
    lidar: L.LidarConfig
    tracker: TrackerConfig
    source: str = "lidar"
    truth_noise_std: float = 0.02
    perception_range: float = 4.0


def geometry_from_omegaconf(model_cfg, env_cfg):
    """None for model.geometry == "ground_truth" (default), else a LidarGeometry."""
    mode = str(model_cfg.get("geometry", "ground_truth"))
    if mode == "ground_truth":
        return None
    if mode not in ("lidar", "tracked_truth"):
        raise ValueError(f"model.geometry must be ground_truth, lidar or tracked_truth, got {mode!r}")
    lidar = L.lidar_config_from_omegaconf(model_cfg.get("lidar", None), env_cfg)
    tracker = tracker_config_from_omegaconf(model_cfg.get("tracker", None), env_cfg)
    if lidar.max_clusters < tracker.num_slots:
        raise ValueError("lidar.max_clusters must be >= num_humans")
    extra = model_cfg.get("lidar", None) or {}
    return LidarGeometry(
        lidar=lidar,
        tracker=tracker,
        source="lidar" if mode == "lidar" else "truth",
        truth_noise_std=float(extra.get("truth_noise_std", 0.02)),
        perception_range=float(extra.get("perception_range", 4.0)),
    )


def _detections(geo: LidarGeometry, state, key):
    if geo.source == "truth":
        return L.noisy_truth_detections(state, geo.lidar.max_clusters, geo.truth_noise_std, key)
    dets, _scan, _labels = L.detect_from_state(geo.lidar, state, key)
    return dets


def lidar_frame_init(env_cfg, geo: LidarGeometry, state, key):
    """First frame of an episode batch: fresh tracker, one scan. Returns (frame, tracker)."""
    batch = state.robot_state.shape[0]
    key = jax.random.fold_in(key, LIDAR_KEY_SALT)
    tracker = tracker_step(geo.tracker, init_tracker(geo.tracker, batch), _detections(geo, state, key),
                           jnp.ones((batch,), bool))
    return lidar_policy_obs(env_cfg, state, tracker, geo.tracker), tracker


def lidar_frame_step(env_cfg, geo: LidarGeometry, tracker, state, key, reset_mask):
    """Scan the (already reset) next state and update the tracker. Returns (frame, tracker).

    reset_mask: envs whose state is the first step of a new episode; their tracks are cleared first.
    """
    key = jax.random.fold_in(key, LIDAR_KEY_SALT)
    tracker = tracker_step(geo.tracker, tracker, _detections(geo, state, key), reset_mask)
    return lidar_policy_obs(env_cfg, state, tracker, geo.tracker), tracker


def init_lidar_obs_history(env_cfg, geo: LidarGeometry, state, history_len, key):
    """Same as trainer.jax_ppo_rollout.init_obs_history, but the frame comes from the tracker."""
    frame, tracker = lidar_frame_init(env_cfg, geo, state, key)
    return jnp.repeat(frame[:, None, :], int(history_len), axis=1), tracker