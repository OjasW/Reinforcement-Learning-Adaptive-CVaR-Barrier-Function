"""Visualise the spatio-temporal graph built by _graph_ in jax_ppo_gnn.py.

Node layout (per timestep t, n = N + 2 nodes):  id = t * n + k
    k = 0       robot
    k = 1..N    humans
    k = N + 1   goal

Edge layout (same order as _edge_index / edge_attr):
    spatial  : (T, N + 1) entries, human k (and goal) -> robot at the same t
    temporal : (T - 1, N + 1) entries, node k at t -> node k at t + 1 (robot, humans)

edge_attr columns: [dx, dy, dvx, dvy, dr, is_spatial, mask]
node_features columns: [is_robot, is_goal, is_human, mask]
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D


def build_edge_index(T: int, N: int):
    """Numpy version of _edge_index, so this file also runs standalone."""
    n = N + 2
    t = np.arange(T)[:, None]
    sp_src = (t * n + np.arange(1, N + 2)[None, :]).reshape(-1)
    sp_dst = np.repeat(np.arange(T) * n, N + 1)
    tt = np.arange(T - 1)[:, None]
    tm_src = (tt * n + np.arange(0, N + 1)[None, :]).reshape(-1)
    tm_dst = tm_src + n
    return np.concatenate([sp_src, tm_src]), np.concatenate([sp_dst, tm_dst])


def _cluster_layout(T: int, N: int, spacing: float = 4.0, radius: float = 1.2):
    """One cluster per timestep. Robot at the centre, humans and goal on a circle."""
    n = N + 2
    pos = {}
    phase = np.pi / 2 + 0.4  # small tilt so nothing sits on the horizontal axis
    for t in range(T):
        cx = t * spacing
        pos[t * n] = (cx, 0.0)
        for s in range(N + 1):  # slot 0 = goal, slots 1..N = humans
            ang = phase + 2 * np.pi * s / (N + 1)
            xy = (cx + radius * np.cos(ang), radius * np.sin(ang))
            node = t * n + (N + 1 if s == 0 else s)
            pos[node] = xy
    return pos


def visualize_graph(
    node_features,
    edge_attr,
    src,
    dst,
    history_len: int,
    num_humans: int,
    b: int = 0,
    show_masked: bool = False,
    label_edges: bool = False,
    save_path: str | None = None,
    title: str | None = None,
):
    """Draw one sample of the batch.

    Args:
        node_features: (B, T*(N+2), 4) from _graph_
        edge_attr:     (B, E, 7) from _graph_
        src, dst:      (E,) static edge index from _graph_
        b:             which batch element to draw
        show_masked:   also draw masked (absent) nodes and edges, faded
        label_edges:   write the norm of the (dx, dy) part on each edge
    """
    T, N = history_len, num_humans
    n = N + 2
    nf = np.asarray(node_features)[b]
    ea = np.asarray(edge_attr)[b]
    src = np.asarray(src)
    dst = np.asarray(dst)
    assert nf.shape[0] == T * n, f"expected {T * n} nodes, got {nf.shape[0]}"
    assert ea.shape[0] == src.shape[0], "edge_attr and edge_index disagree"

    node_valid = nf[:, 3] > 0.5
    is_spatial = ea[:, 5] > 0.5
    edge_valid = ea[:, 6] > 0.5

    g = nx.DiGraph()
    g.add_nodes_from(range(T * n))
    pos = _cluster_layout(T, N)

    fig, ax = plt.subplots(figsize=(max(6.0, 4.2 * T), 5.5))

    # ---- edges ----
    def draw_edges(mask, color, rad, alpha, style):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        edgelist = [(int(src[i]), int(dst[i])) for i in idx]
        nx.draw_networkx_edges(
            g, pos, edgelist=edgelist, ax=ax, edge_color=color, alpha=alpha,
            style=style, arrows=True, arrowstyle="-|>", arrowsize=9, width=1.2,
            connectionstyle=f"arc3,rad={rad}", node_size=600,
        )

    for valid_flag, alpha, style in [(True, 0.75, "solid"), (False, 0.15, "dashed")]:
        if not valid_flag and not show_masked:
            continue
        sel = edge_valid == valid_flag
        draw_edges(sel & is_spatial, "tab:gray", 0.0, alpha, style)
        draw_edges(sel & ~is_spatial, "tab:orange", 0.25, alpha, style)

    if label_edges:
        for i in range(len(src)):
            if not (edge_valid[i] or show_masked):
                continue
            (x0, y0), (x1, y1) = pos[int(src[i])], pos[int(dst[i])]
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{np.linalg.norm(ea[i, :2]):.2f}",
                    fontsize=6, color="black", ha="center", va="center")

    # ---- nodes ----
    robot = [t * n for t in range(T)]
    humans = [t * n + k for t in range(T) for k in range(1, N + 1)]
    goal = [t * n + N + 1 for t in range(T)]
    for nodes, color, size in [(robot, "red", 700), (humans, "lightblue", 500), (goal, "green", 700)]:
        ok = [v for v in nodes if node_valid[v]]
        bad = [v for v in nodes if not node_valid[v]]
        nx.draw_networkx_nodes(g, pos, nodelist=ok, node_size=size, node_color=color,
                               edgecolors="black", linewidths=0.8, ax=ax)
        if show_masked and bad:
            nx.draw_networkx_nodes(g, pos, nodelist=bad, node_size=size, node_color="white",
                                   edgecolors="gray", linewidths=0.8, alpha=0.5, ax=ax)

    labels = {}
    for t in range(T):
        labels[t * n] = "R"
        for k in range(1, N + 1):
            labels[t * n + k] = f"H{k}"
        labels[t * n + N + 1] = "G"
    shown = {v: s for v, s in labels.items() if node_valid[v] or show_masked}
    nx.draw_networkx_labels(g, pos, labels=shown, font_size=8, ax=ax)

    # ---- time headers ----
    for t in range(T):
        if t == T - 1:
            tag = "t (latest, readout R)"
        else:
            tag = f"t-{T - 1 - t}"
        ax.text(t * 4.0, 2.0, tag, ha="center", fontsize=10, fontweight="bold")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markeredgecolor="k", markersize=10, label="robot"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightblue", markeredgecolor="k", markersize=10, label="human"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markeredgecolor="k", markersize=10, label="goal"),
        Line2D([0], [0], color="tab:gray", lw=2, label="spatial edge"),
        Line2D([0], [0], color="tab:orange", lw=2, label="temporal edge"),
    ]
    ax.legend(handles=legend, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.0), frameon=False)

    n_valid_nodes = int(node_valid.sum())
    n_valid_edges = int(edge_valid.sum())
    ax.set_title(title or f"Spatio-temporal graph (sample {b}): {n_valid_nodes} valid nodes, "
                          f"{n_valid_edges}/{len(src)} valid edges")
    ax.set_ylim(-3.2, 2.5)  # leave room for curved temporal arcs and time labels
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # Demo with random data in exactly the layout _graph_ returns.
    import sys
    

    T, N, B = 3, 5, 2
    n = N + 2
    rng = np.random.default_rng(0)

    type_onehot = np.zeros((B, T, n, 3), dtype=np.float32)
    type_onehot[:, :, 0, 0] = 1
    type_onehot[:, :, -1, 1] = 1
    type_onehot[:, :, 1:-1, 2] = 1
    human_mask = (rng.random((B, T, N, 1)) > 0.3).astype(np.float32)
    ones = np.ones((B, T, 1, 1), dtype=np.float32)
    node_features = np.concatenate(
        [type_onehot, np.concatenate([ones, human_mask, ones], axis=2)], axis=-1
    ).reshape(B, T * n, 4)

    # spatial (B,T,N+1,7) then temporal (B,T-1,N+1,7)
    sp = rng.normal(size=(B, T, N + 1, 7)).astype(np.float32)
    sp[..., 5] = 1
    sp[..., 6] = np.concatenate([human_mask[..., 0], np.ones((B, T, 1))], axis=2)
    tm = rng.normal(size=(B, T - 1, N + 1, 7)).astype(np.float32)
    tm[..., 5] = 0
    m2 = human_mask[:, :-1] * human_mask[:, 1:]
    tm[..., 6] = np.concatenate([np.ones((B, T - 1, 1)), m2[..., 0]], axis=2)
    edge_attr = np.concatenate([sp.reshape(B, -1, 7), tm.reshape(B, -1, 7)], axis=1)

    src, dst = build_edge_index(T, N)
    out = sys.argv[1] if len(sys.argv) > 1 else "graph_demo.png"
    visualize_graph(node_features, edge_attr, src, dst, T, N, b=0, save_path=out)
    print(f"saved {out}")
