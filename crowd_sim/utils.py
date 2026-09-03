import numpy as np
 

def is_absolute_obs_dim(obs_dim: int) -> bool:
    dim = int(obs_dim)
    return dim >= 8 and (dim - 8) % 6 == 0


def relative_obs_dim_from_env_dim(obs_dim: int) -> int:
    dim = int(obs_dim)
    if is_absolute_obs_dim(dim):
        # abs format: 8 + K*6 -> relative policy format: 6 + K*6
        return dim - 2
    return dim


def absolute_obs_to_relative(obs):
    """
    Convert observation from absolute format to relative policy format.

    Absolute format (1D):
      [rx, ry, gx, gy, rvx, rvy, rtheta, rr, (hx, hy, hvx, hvy, hr, mask)*K]

    Relative policy format (1D):
      [goal_rel_x, goal_rel_y, rvx, rvy, rtheta, rr, (rel_x, rel_y, hvx, hvy, hr, mask)*K]
    """
    x = np.asarray(obs, dtype=np.float32).reshape(-1)

    # Pass through already-relative observations.
    if x.size >= 6 and (x.size - 6) % 6 == 0:
        if not (x.size >= 8 and (x.size - 8) % 6 == 0):
            return x
        # If both checks pass (unlikely/ambiguous), prefer absolute interpretation.

    if not (x.size >= 8 and (x.size - 8) % 6 == 0):
        raise ValueError(f"Unsupported observation length for abs->rel conversion: {x.size}")

    k = (x.size - 8) // 6
    out = np.zeros((6 + 6 * k,), dtype=np.float32)

    rx, ry, gx, gy, rvx, rvy, rtheta, rr = x[:8]
    out[0] = rx - gx
    out[1] = ry - gy
    out[2] = rvx
    out[3] = rvy
    out[4] = rtheta
    out[5] = rr

    if k > 0:
        blocks = x[8:].reshape(k, 6)
        out_blocks = np.zeros((k, 6), dtype=np.float32)
        out_blocks[:, 0] = rx - blocks[:, 0]
        out_blocks[:, 1] = ry - blocks[:, 1]
        out_blocks[:, 2:6] = blocks[:, 2:6]
        out[6:] = out_blocks.reshape(-1)

    return out


def absolute_obs_batch_to_relative(obs_batch):
    """
    Batch version of absolute_obs_to_relative.
    Input can be shape (N, D) absolute observations or already-relative (N, D_rel).
    """
    arr = np.asarray(obs_batch, dtype=np.float32)

    if arr.ndim == 1:
        return absolute_obs_to_relative(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected obs batch with ndim 1 or 2, got shape {arr.shape}")

    n, d = arr.shape
    if d >= 8 and (d - 8) % 6 == 0:
        k = (d - 8) // 6
        out = np.zeros((n, 6 + 6 * k), dtype=np.float32)

        rx = arr[:, 0:1]
        ry = arr[:, 1:2]
        gx = arr[:, 2:3]
        gy = arr[:, 3:4]

        out[:, 0:1] = rx - gx
        out[:, 1:2] = ry - gy
        out[:, 2:6] = arr[:, 4:8]

        if k > 0:
            blocks = arr[:, 8:].reshape(n, k, 6)
            out_blocks = np.zeros((n, k, 6), dtype=np.float32)
            out_blocks[:, :, 0] = rx - blocks[:, :, 0]
            out_blocks[:, :, 1] = ry - blocks[:, :, 1]
            out_blocks[:, :, 2:6] = blocks[:, :, 2:6]
            out[:, 6:] = out_blocks.reshape(n, 6 * k)

        return out

    if d >= 6 and (d - 6) % 6 == 0:
        return arr

    raise ValueError(f"Unsupported batch observation width for abs->rel conversion: {d}")


def select_top_k_obs(rel_obs, top_k: int):
    """Slice the first ``top_k`` human blocks from a relative-format obs.

    The env emits all humans (sorted nearest-first) so a simple prefix slice
    is the correct top-K selection. Works on 1D or 2D arrays.

    Input shape:  (..., 6 + N_all * 6)
    Output shape: (..., 6 + top_k * 6)
    """
    k = int(top_k)
    arr = np.asarray(rel_obs)
    width = 6 + k * 6
    if arr.shape[-1] < width:
        raise ValueError(
            f"obs has {arr.shape[-1]} cols, need >= {width} for top_k={k}"
        )
    if arr.shape[-1] == width:
        return arr
    return arr[..., :width]


def sample_point_in_disk(rng, center, radius, arena_size=None, max_tries=256):
    center = np.asarray(center, dtype=float)
    for _ in range(max_tries):
        rr = radius * np.sqrt(rng.uniform(0.0, 1.0))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        p = center + rr * np.array([np.cos(theta), np.sin(theta)], dtype=float)
        if arena_size is None:
            return p
        if (-arena_size <= p[0] <= arena_size) and (-arena_size <= p[1] <= arena_size):
            return p
    if arena_size is None:
        return center.copy()
    return np.clip(center, -arena_size, arena_size)

def relative_obs_to_graph(obs):
    """
    Convert a relative-format observation to a graph representation.

    Relative format:
      [robot(6), human_1(6), ..., human_N(6)]

    Returns:
        robot:  (6,)
        humans: (N, 6)

    The final human feature is the validity mask. --->wdym?
    """

    arr = np.asarray(obs, dtype=np.float32)

    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D relative observation, got shape {arr.shape}"
        )

    if arr.size < 6 or (arr.size - 6) % 6 != 0:
        raise ValueError(
            f"Invalid relative observation dimension: {arr.size}"
        )

    robot = arr[:6]
    humans = arr[6:].reshape(-1, 6)

    return robot, humans

def relative_obs_to_graph(obs):
    """
    Convert relative flattened observation into robot/human graph tensors.

    Relative observation:
        [robot(6), human_1(6), ..., human_N(6)]

    Returns:
        robot:  (6,)
        humans: (N, 6)

    The final human feature is the validity mask.
    """
    arr = np.asarray(obs, dtype=np.float32)

    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D relative observation, got shape {arr.shape}"
        )

    if arr.size < 6 or (arr.size - 6) % 6 != 0:
        raise ValueError(
            f"Invalid relative observation dimension: {arr.size}"
        )

    robot = arr[:6]
    humans = arr[6:].reshape(-1, 6)

    return robot, humans


def relative_obs_batch_to_graph(obs_batch):
    """
    Convert batched relative observations into graph components.

    Input:
        (B, 6 + N*6)

    Returns:
        robot:
            (B, 6)

        humans:
            (B, N, 6)
    """
    arr = np.asarray(obs_batch, dtype=np.float32)

    if arr.ndim == 1:
        robot, humans = relative_obs_to_graph(arr)
        return robot[None, :], humans[None, :, :]

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D observation batch, got {arr.shape}"
        )

    n, d = arr.shape

    if d < 6 or (d - 6) % 6 != 0:
        raise ValueError(
            f"Invalid relative observation width: {d}"
        )

    robot = arr[:, :6]
    humans = arr[:, 6:].reshape(n, -1, 6)

    return robot, humans


def build_env(env_name: str, render_mode: str, config):
    from crowd_sim.env.social_nav import SocialNav
    from crowd_sim.env.social_nav_var_num import SocialNavVarNum

    if env_name == "social_nav":
        return SocialNav(render_mode=render_mode, config_file=config)
    if env_name == "social_nav_var_num":
        return SocialNavVarNum(render_mode=render_mode, config_file=config)
    raise ValueError(f"Unknown env: {env_name}")


def resolve_env_name(config) -> str:
    return str(config.env.get("name", "social_nav_var_num"))
