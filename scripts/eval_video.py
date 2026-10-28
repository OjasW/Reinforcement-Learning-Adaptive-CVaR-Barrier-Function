"""Video helpers for scripts/eval.py.

[SPEED] Moved here unchanged from scripts/eval.py (render_jax_social_nav_frame, save_video and helpers), so
worker processes can import them and draw video frames in parallel. eval.py re-exports the same names.
[IMG] Adds the optional --policy-view panel: the exact image stack ppo_cnn received at each step, shown next
to the arena view, plus a dashed square in the arena marking the CNN's image window.

Only numpy / matplotlib / Pillow here: no JAX, so worker processes start fast and never touch the GPU.
"""

import numpy as np


def _frame_to_rgb_array(frame):
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def save_video(path, frames, fps=30):
    import imageio

    frames = [_frame_to_rgb_array(frame) for frame in frames]
    imageio.mimsave(path, frames, fps=fps)


def _scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def render_jax_social_nav_frame(
    env_cfg,
    state,
    total_return=0.0,
    status="",
    size=740,
    robot_traj=None,
    qp_viz=None,
    policy_window=None,  # [IMG] --policy-view: (4, 2) world-frame corners of the CNN image window
):
    """Render one JAX SocialNav state as a Matplotlib frame."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    robot_state = np.asarray(state.robot_state[0], dtype=float)
    robot_pos = robot_state[:2]
    robot_theta = float(robot_state[2]) if robot_state.shape[0] > 2 else 0.0
    robot_radius = float(env_cfg.robot_radius)
    human_positions = np.asarray(state.human_positions[0], dtype=float)
    human_radii = np.asarray(state.human_radii[0], dtype=float)
    human_mask = np.asarray(state.human_mask[0], dtype=float) > 0.5
    goal_pos = np.asarray(state.goal_pos[0], dtype=float)

    robot_traj_arr = np.asarray(robot_traj if robot_traj is not None else [robot_pos], dtype=float)
    start_pos = np.asarray(robot_traj_arr[0] if robot_traj_arr.size else robot_pos, dtype=float)

    low = np.asarray(getattr(env_cfg, "workspace_low", [-6.0, -6.0]), dtype=float)
    high = np.asarray(getattr(env_cfg, "workspace_high", [6.0, 6.0]), dtype=float)
    boundary_buffer = max(float(getattr(env_cfg, "boundary_buffer", 0.0)), 0.0)
    plot_low = low - boundary_buffer
    plot_high = high + boundary_buffer
    plot_span = np.maximum(plot_high - plot_low, 1e-6)
    plot_aspect = float(plot_span[0] / plot_span[1])

    frame_width = max(16, int(np.ceil(float(size) / 16.0)) * 16)
    frame_height = max(16, int(np.ceil((frame_width / plot_aspect) / 16.0)) * 16)
    dpi = max(80, int(size / 7.4))
    fig = Figure(figsize=(frame_width / dpi, frame_height / dpi), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.13, top=0.90)

    ax.clear()
    ax.set_xlim(float(plot_low[0]), float(plot_high[0]))
    ax.set_ylim(float(plot_low[1]), float(plot_high[1]))
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    robot_color = "#3b82f6"
    traj_cmap = LinearSegmentedColormap.from_list("robot_traj", ["#dbeafe", robot_color])
    qp_viz = qp_viz or {}
    robot_type = str(getattr(env_cfg, "robot_type", "unicycle"))
    lookahead_distance = float(qp_viz.get("lookahead_distance", 0.0) or 0.0)
    if robot_type != "unicycle":
        lookahead_distance = 0.0

    topk_ids = np.asarray(qp_viz.get("topk_ids", []), dtype=int).reshape(-1)
    r_safe = np.asarray(qp_viz.get("r_safe", []), dtype=float).reshape(-1)
    beta = np.asarray(qp_viz.get("beta", []), dtype=float).reshape(-1)
    beta_min = float(qp_viz.get("beta_min", 0.05) or 0.05)
    beta_total = float(
        qp_viz.get("beta_total", qp_viz.get("beta_max", 0.5)) or 0.5
    )
    active_topk_count = sum(
        0 <= human_id < human_positions.shape[0]
        and (human_id >= human_mask.shape[0] or bool(human_mask[human_id]))
        for human_id in topk_ids
    )
    remaining_beta_budget = max(beta_total - active_topk_count * beta_min, 1e-6)
    beta_individual_max = beta_min + remaining_beta_budget
    base_human_color = np.asarray(to_rgba("#8c8c8c"), dtype=float)
    risky_human_color = np.asarray(to_rgba("#dc2626"), dtype=float)
    human_colors = np.repeat(base_human_color[None, :], human_positions.shape[0], axis=0)
    for rank, human_id in enumerate(topk_ids):
        if human_id < 0 or human_id >= human_positions.shape[0] or rank >= r_safe.shape[0]:
            continue
        if human_id < human_mask.shape[0] and not human_mask[human_id]:
            continue
        beta_i = (
            float(beta[rank])
            if rank < beta.shape[0] and np.isfinite(beta[rank])
            else beta_individual_max
        )
        beta_t = np.clip(
            (beta_i - beta_min) / remaining_beta_budget, 0.0, 1.0
        )
        human_colors[human_id] = risky_human_color * (1.0 - beta_t) + base_human_color * beta_t
        adaptive_radius = float(r_safe[rank]) - robot_radius
        if not np.isfinite(adaptive_radius):
            continue
        adaptive_radius = max(adaptive_radius, float(human_radii[human_id]))
        ax.add_artist(
            Circle(
                human_positions[human_id],
                adaptive_radius,
                facecolor=to_rgba("#9ca3af", 0.16),
                edgecolor="none",
                linewidth=0.0,
                zorder=1.7,
            )
        )

    for i, pos in enumerate(human_positions):
        if i < human_mask.shape[0] and not human_mask[i]:
            continue
        ax.add_artist(Circle(pos, human_radii[i], facecolor=human_colors[i], edgecolor="none", linewidth=0.0, alpha=0.95, zorder=2))

    if robot_traj_arr.shape[0] > 1:
        points = robot_traj_arr.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_values = np.linspace(0.0, 1.0, len(segments), dtype=float)
        collection = LineCollection(
            segments,
            cmap=traj_cmap,
            norm=Normalize(vmin=0.0, vmax=1.0),
            linewidths=4.4,
            alpha=0.98,
            zorder=4,
        )
        collection.set_array(seg_values)
        collection.set_capstyle("round")
        ax.add_collection(collection)

    ax.add_artist(Circle(robot_pos, robot_radius, facecolor=robot_color, edgecolor="none", linewidth=0.0, alpha=0.82, zorder=6))
    if lookahead_distance > 0.0:
        lookahead_point = robot_pos + lookahead_distance * np.asarray([np.cos(robot_theta), np.sin(robot_theta)], dtype=float)
        ax.add_artist(
            Circle(
                lookahead_point,
                robot_radius + lookahead_distance,
                facecolor="none",
                edgecolor="#1d4ed8",
                linewidth=1.4,
                linestyle="--",
                alpha=0.7,
                zorder=7,
            )
        )
        ax.plot([robot_pos[0], lookahead_point[0]], [robot_pos[1], lookahead_point[1]], color="#1d4ed8", linewidth=1.2, alpha=0.85, zorder=8)
        ax.scatter([lookahead_point[0]], [lookahead_point[1]], s=56, c="#1d4ed8", edgecolors="white", linewidths=0.8, zorder=9)

    ax.scatter([start_pos[0]], [start_pos[1]], s=420, marker="^", c="#ead94c", edgecolors="none", linewidths=0.0, zorder=8)
    ax.add_artist(Circle(goal_pos, 0.3, facecolor="#ead94c", edgecolor="#d2bc38", linewidth=1.2, alpha=0.65, zorder=7))

    start_handle = Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="#ead94c", markeredgecolor="none", markersize=9, label="Start")
    goal_handle = Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#ead94c", markeredgecolor="#d2bc38", markeredgewidth=1.0, markersize=9, label="Goal")
    traj_handle = Line2D([0], [0], color=robot_color, linewidth=4.4, alpha=0.98, label="Trajectory")
    handles = [start_handle, goal_handle, traj_handle]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=13, borderpad=0.3, handletextpad=0.4)

    ax.grid(False)
    ax.tick_params(labelsize=18, colors="black")
    current_time = float(_scalar(state.step_count[0])) * float(env_cfg.dt)
    title = f"Time: {current_time:.1f}s"
    if status:
        title += f" | {status}"
    ax.set_title(title, fontsize=20, pad=7, color="black")

    if policy_window is not None:  # [IMG] dashed square = the area the CNN can see this step
        from matplotlib.patches import Polygon
        ax.add_patch(Polygon(np.asarray(policy_window, dtype=float), closed=True, fill=False,
                             edgecolor="#16a34a", linewidth=1.6, linestyle="--", alpha=0.9, zorder=10))

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


# ----------------------------------------------------------------------------- [IMG] policy view
# Same colours as scripts/check_policy_images.py: white background, blue robot, green goal, red humans.
POLICY_VIEW_COLORS = np.array([[0.20, 0.45, 1.00],   # ch0 robot
                               [0.10, 0.80, 0.20],   # ch1 goal
                               [1.00, 0.25, 0.20]])  # ch2 humans


def policy_frame_to_rgb(img_u8):
    """One (H, W, 3) uint8 policy image (channels = robot, goal, humans) -> RGB uint8 for display."""
    x = np.asarray(img_u8, dtype=np.float32) / 255.0
    rgb = 1.0 - np.einsum("hwc,ck->hwk", x, 1.0 - POLICY_VIEW_COLORS)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def compose_policy_view(arena_rgb, stack_u8):
    """Arena frame (Ha, Wa, 3) + a panel with the T policy images (T, H, W, 3), oldest -> newest.

    Tiles are scaled up with nearest-neighbour (no smoothing), so every pixel is exactly what the CNN saw.
    Total width is padded to a multiple of 16 for the video encoder.
    """
    from PIL import Image, ImageDraw

    arena_rgb = np.asarray(arena_rgb, dtype=np.uint8)
    stack_u8 = np.asarray(stack_u8, dtype=np.uint8)
    t_len, img_h = int(stack_u8.shape[0]), int(stack_u8.shape[1])
    ha, wa = arena_rgb.shape[:2]
    cols = 2
    rows = (t_len + cols - 1) // cols
    pad, title_h, label_h = 10, 34, 24
    scale = max(1, (ha - title_h - rows * label_h - (rows + 1) * pad) // (rows * img_h))
    tile = img_h * scale
    panel_w = cols * tile + (cols + 1) * pad
    panel_w += (-(wa + panel_w)) % 16

    panel = Image.new("RGB", (panel_w, ha), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((pad, 8), "CNN input (what the policy sees)", fill="black", font=_font(18))
    for i in range(t_len):
        r, c = divmod(i, cols)
        x0 = pad + c * (tile + pad)
        y0 = title_h + pad + r * (tile + label_h + pad)
        age = t_len - 1 - i
        label = "t (now)" if age == 0 else f"t-{age}"
        draw.text((x0, y0), label, fill="black", font=_font(15))
        tile_img = Image.fromarray(policy_frame_to_rgb(stack_u8[i])).resize((tile, tile), Image.NEAREST)
        panel.paste(tile_img, (x0, y0 + label_h))
        draw.rectangle([x0 - 1, y0 + label_h - 1, x0 + tile, y0 + label_h + tile], outline="#9ca3af")
    return np.concatenate([arena_rgb, np.asarray(panel, dtype=np.uint8)], axis=1)


# ----------------------------------------------------------------------------- [SPEED] parallel frame drawing
_WORKER_ENV = None


def init_render_worker(env_lite):
    """Pool initializer: keep the (small, picklable) env settings in each worker once."""
    global _WORKER_ENV
    _WORKER_ENV = env_lite


def render_job(job):
    """Draw one video frame. job = (env_lite or None, state, total, status, robot_traj, qp_viz, window, stack)."""
    env_lite, state, total, status, robot_traj, qp_viz, window, stack = job
    env_lite = env_lite if env_lite is not None else _WORKER_ENV
    frame = render_jax_social_nav_frame(env_lite, state, total, status=status, robot_traj=robot_traj,
                                        qp_viz=qp_viz, policy_window=window)
    if stack is not None:
        frame = compose_policy_view(frame, stack)
    return frame
