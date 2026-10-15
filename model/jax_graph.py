import jax
import jax.numpy as jnp


def _graph_(obs, hparams):
    obs = jnp.asarray(obs, dtype=jnp.float32)
    if obs.ndim != 2:
        raise ValueError(f"ppo_gnn expects rank-2 observations, got ndim={obs.ndim}")
    if obs.shape[-1] != hparams.obs_dim:
        raise ValueError(f"ppo_gnn expects obs_dim={hparams.obs_dim}, got {obs.shape[-1]}")

    B, T, N = obs.shape[0], hparams.history_len, hparams.num_humans

    frames = obs.reshape((B, T, hparams.obs_frame_dim))
    robot_theta = jnp.broadcast_to(frames[:, -1:, 4], (B, T)) # changed from frames[:, :, 4]
    goal_rel = _world_to_body(frames[:, :, 0:2], robot_theta) / hparams.position_scale
    robot_vel = _world_to_body(frames[:, :, 2:4], robot_theta) / hparams.velocity_scale
    robot_radius = frames[:, :, 5:6] / hparams.position_scale

    human_blocks = frames[:, :, 6:].reshape(
        (B, T, N, 6)
    )
    human_mask = jnp.clip(human_blocks[:, :, :, 5:6], 0.0, 1.0)
    human_rel = _world_to_body(human_blocks[:, :, :, 0:2], robot_theta) / hparams.position_scale
    human_vel = _world_to_body(human_blocks[:, :, :, 2:4], robot_theta) / hparams.velocity_scale
    human_radius = human_blocks[:, :, :, 4:5] / hparams.position_scale
    human_values = jnp.concatenate([human_rel, robot_vel[:, :, None, :] - human_vel, human_radius], axis=-1) * human_mask

    # ------------ deltas --------------- #
    goal_delta = goal_rel[:, 1:] - goal_rel[:, :-1]
    robot_delta = -goal_delta
    robot_vel_d = robot_vel[:, 1:] - robot_vel[:, :-1]
    robot_rad_d = robot_radius[:, 1:] - robot_radius[:, :-1]

    m2 = human_mask[:, :-1] * human_mask[:, 1:]    # (B,T-1,N,1)
    human_disp = (human_rel[:, 1:] - human_rel[:, :-1] - goal_delta[:, :, None, :]) * m2
    human_vel_d = (human_vel[:, 1:] - human_vel[:, :-1]) * m2
    human_rad_d = (human_radius[:, 1:] - human_radius[:, :-1]) * m2

    # ------------ nodes --------------- #
    type_onehot = jnp.zeros((B, T, N + 2, 3), dtype=jnp.float32)
    type_onehot = type_onehot.at[:, :, 0, 0].set(1.0)  # robot node
    type_onehot = type_onehot.at[:, :, -1, 1].set(1.0)  # goal node
    type_onehot = type_onehot.at[:, :, 1:-1, 2].set(1.0)  # human nodes
    ones = jnp.ones((B, T, 1, 1), dtype=jnp.float32)
    all_mask = jnp.concatenate([ones, human_mask, ones], axis=2)
    node_features = jnp.concatenate([type_onehot, all_mask], axis=-1)

    # ------------ edges --------------- #
    # spatial = [human_rel(2), human_vel_rel(2), human_radius_rel(1), spatial_onehot(1), mask=1(1);
    #               goal_rel(2), goal_vel_rel(2), 0(1), spatial_onehot=1(1), mask=1(1)]
    ones_s = jnp.ones((B, T, 1), dtype=jnp.float32)
    goal_spatial_edges = jnp.concatenate([goal_rel, robot_vel, jnp.zeros((B, T, 1), dtype=jnp.float32), ones_s, ones_s], axis=-1)                     # (B,T,7)
    human_spatial_edges = jnp.concatenate([human_values, jnp.ones((B, T, N, 1), dtype=jnp.float32), human_mask], axis=-1)                              # (B,T,N,7)
    spatial_edges = jnp.concatenate([human_spatial_edges, goal_spatial_edges[:, :, None, :]], axis=2)                               # (B,T,N+1,7)
    
    # temporal_edges = [robot_delta=goal_delta(2), robot_vel_d(2), robot_rad_d(1), temporal_onehot=0(1), mask=1(1);
    #                   [human_delta(2), human_vel_d(2), human_rad_d(1), temporal_onehot=0(1), mask=1(1)]
    robot_temporal = jnp.concatenate([robot_delta, robot_vel_d, robot_rad_d, jnp.zeros((B, T-1, 1), dtype=jnp.float32), jnp.ones((B, T-1, 1), dtype=jnp.float32)], axis=-1)                                     # (B,T-1,7)
    human_temporal = jnp.concatenate([human_disp, human_vel_d, human_rad_d, jnp.zeros((B, T-1, N, 1), dtype=jnp.float32), m2], axis=-1)                                                      # (B,T-1,N,7)
    temporal_edges = jnp.concatenate([robot_temporal[:, :, None, :], human_temporal], axis=2)                                                                             # (B,T-1,N+1,7)

    node_features = node_features.reshape((B, T * (N + 2), 4))
    edge_attr = jnp.concatenate([spatial_edges.reshape(B, -1, 7), temporal_edges.reshape(B, -1, 7)], axis=1)
    src, dst = _edge_index(T, N)
    assert edge_attr.shape[1] == src.shape[0]
    robot_last = (T - 1) * (N + 2)  # index of the last robot node in the flattened node_features

    return node_features, edge_attr, src, dst, robot_last

def _edge_index(T, N):
    n = N + 2
    t = jnp.arange(T)[:, None]
    sp_src = (t * n + jnp.arange(1, n)[None, :]).reshape(-1)
    sp_dst = jnp.repeat(jnp.arange(T) * n, n - 1)
    tt = jnp.arange(T - 1)[:, None]
    tm_src = (tt * n + jnp.arange(0, N + 1)[None, :]).reshape(-1)
    tm_dst = tm_src + n
    src = jnp.concatenate([sp_src, tm_src])
    dst = jnp.concatenate([sp_dst, tm_dst])
    return jnp.asarray(src, dtype=jnp.int32), jnp.asarray(dst, dtype=jnp.int32)

def _world_to_body(vectors, robot_theta):
    c = jnp.cos(robot_theta)
    s = jnp.sin(robot_theta)
    while c.ndim < vectors.ndim - 1:
        c = c[..., None]
        s = s[..., None]
    x = vectors[..., 0]
    y = vectors[..., 1]
    return jnp.stack([c * x + s * y, -s * x + c * y], axis=-1)
