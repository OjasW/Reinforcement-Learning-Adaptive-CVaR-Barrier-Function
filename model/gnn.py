from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_graph_from_rel_obs(obs: torch.Tensor, max_humans: int):
    """
    obs: (B, 6 + max_humans * 6) relative-format observation, as produced
         by absolute_obs_to_relative / absolute_obs_batch_to_relative.
         

    Returns
    -------
    node_feats: (B, N, 8)  N = 1 (robot) + max_humans + 1 (goal); robot is index 0.
        Per-node layout: [pos_x, pos_y, vel_x, vel_y, radius, cos_h, sin_h, valid]
        pos/vel are expressed relative to the robot (robot pos = 0, vel = 0).
    node_mask: (B, N) bool, True = valid/present node.
    robot_idx: int, always 0.
    """

    if obs.dim() != 2:
        raise ValueError(
            f"Expected obs with shape (B, D), got {tuple(obs.shape)}"
        )

    B, D = obs.shape
    expected_dim = 6 + 6 * max_humans

    if D != expected_dim:
        raise ValueError(
            f"Expected observation dimension {expected_dim} "
            f"for max_humans={max_humans}, got {D}"
        )

    goal_rel_x, goal_rel_y = obs[:, 0], obs[:, 1]
    rvx, rvy, rtheta, rr = obs[:, 2], obs[:, 3], obs[:, 4], obs[:, 5]

    blocks = obs[:, 6:].reshape(B, max_humans, 6)
    rel_x, rel_y = blocks[..., 0], blocks[..., 1]
    hvx, hvy = blocks[..., 2], blocks[..., 3]
    hr = blocks[..., 4]
    hmask = blocks[..., 5].clamp(0.0, 1.0)

    zeros = torch.zeros_like(rvx)
    ones = torch.ones_like(rvx)

    # ---- robot node (self-relative: pos=0, vel=0; carries its own state) ----
    robot_feat = torch.stack(
        [zeros, zeros, zeros, zeros, rr, torch.cos(rtheta), torch.sin(rtheta), ones],
        dim=-1,
    )  # (B, 8)

    # ---- goal node: static, so its velocity relative to robot is -robot_vel ----
    goal_pos_x = -goal_rel_x   # gx - rx
    goal_pos_y = -goal_rel_y   # gy - ry
    goal_feat = torch.stack(
        [goal_pos_x, goal_pos_y, -rvx, -rvy, zeros, zeros, zeros, ones],
        dim=-1,
    )  # (B, 8)

    # ---- obstacle nodes: pos relative to robot, vel relative to robot ----
    obs_pos_x = -rel_x   # hx - rx
    obs_pos_y = -rel_y   # hy - ry
    obs_vel_x = hvx - rvx.unsqueeze(1)
    obs_vel_y = hvy - rvy.unsqueeze(1)
    obs_feat = torch.stack(
        [obs_pos_x, obs_pos_y, obs_vel_x, obs_vel_y, hr,
         torch.zeros_like(hr), torch.zeros_like(hr), hmask],
        dim=-1,
    )  # (B, max_humans, 8)

    node_feats = torch.cat(
        [robot_feat.unsqueeze(1), obs_feat, goal_feat.unsqueeze(1)], dim=1
    )  # (B, 1 + max_humans + 1, 8)

    node_mask = torch.cat(
        [torch.ones(B, 1, dtype=torch.bool, device=obs.device),
         hmask.bool(),
         torch.ones(B, 1, dtype=torch.bool, device=obs.device)],
        dim=1,
    )  # (B, N)

    return node_feats, node_mask, 0


def build_edge_feats(node_feats: torch.Tensor):
    """
    Pairwise relative edge features [dx, dy, dvx, dvy, dist] for every
    (i, j) node pair, built from the already robot-relative pos/vel fields.

    node_feats: (B, N, 8) with dims 0:2 = pos, 2:4 = vel
    Returns: (B, N, N, 5)
    """
    pos = node_feats[..., 0:2]
    vel = node_feats[..., 2:4]

    dpos = pos.unsqueeze(2) - pos.unsqueeze(1)   # (B, N, N, 2), edge i->j
    dvel = vel.unsqueeze(2) - vel.unsqueeze(1)   # (B, N, N, 2)
    dist = torch.linalg.norm(dpos, dim=-1, keepdim=True)

    return torch.cat([dpos, dvel, dist], dim=-1)  # (B, N, N, 5)


# --------------------------------------------------------------------------
# 2. Masked multi-head Graph Attention layer (dense, GATv2-style scoring)
# --------------------------------------------------------------------------

class DenseGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, n_heads=4, dropout=0.0):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_msg = nn.Linear(edge_dim, out_dim, bias=False)
        # Edge projections
        self.lin_edge_attn = nn.Linear(edge_dim, out_dim, bias=False)
        self.lin_edge_msg = nn.Linear(edge_dim, out_dim, bias=False)

        self.attn = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)


        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # Residual projection when input/output dimensions differ.
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x, edge_feats, node_mask):
        """
        x:          (B, N, in_dim)
        edge_feats: (B, N, N, edge_dim), edge_feats[b, i, j] = feature of edge i->j
        node_mask:  (B, N) bool
        returns:    (B, N, out_dim)
        """
        B, N, _ = x.shape
        H, D = self.n_heads, self.head_dim

        src = self.lin_src(x).view(B, N, H, D)
        dst = self.lin_dst(x).view(B, N, H, D)
        msg = self.lin_msg(edge_feats).view(B, N, N, H, D)

        # Project edges
        edge_attn = self.lin_edge_attn(edge_feats).view(B, N, N, H, D)
        edge_msg = self.lin_edge_msg(edge_feats).view(B, N, N, H, D) #unnecessary

        #m = dst.unsqueeze(1) + edge                                   # (B, N_i, N_j, H, D)
        pair = (src.unsqueeze(2) + dst.unsqueeze(1) + edge_attn)
        #e = self.leaky((m * self.attn.view(1, 1, 1, H, D)).sum(-1))   # (B, N_i, N_j, H)
        attn_logits = self.leaky_relu((pair * self.attn.view(1, 1, 1, H, D)).sum(dim=-1))

        pad_mask = (~node_mask).view(B, 1, N, 1)
        #e = e.masked_fill(pad_mask, float("-inf"))
        attn_logits = attn_logits.masked_fill(pad_mask, torch.finfo(attn_logits.dtype).min)

        alpha = F.softmax(attn_logits, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = self.dropout(alpha)

        messages = msg + edge_msg

        out = (alpha.unsqueeze(-1) * messages).sum(dim=2)                         # (B, N_i, H, D)
        out = out.reshape(B, N, H * D)
        out = self.out_proj(out)
        residual = self.residual_proj(x)
        out = self.norm(out + residual)

        out = out * node_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out


class GraphAttentionEncoder(nn.Module):
    """
    Full encoder: relative-format obs -> fixed-size robot-node embedding.

    `embed_dim` is fed straight into the existing policy head (fc1) in
    place of the old flattened top-K obs vector.
    """

    def __init__(self, max_humans, node_in_dim=8, edge_in_dim=5,
                 hidden_dim=64, embed_dim=128, n_layers=2, n_heads=4, dropout=0.0):
        super().__init__()
        self.max_humans = max_humans
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)

        dims = [hidden_dim] * n_layers + [embed_dim]
        self.layers = nn.ModuleList([
            DenseGATLayer(dims[i], dims[i + 1], edge_in_dim, n_heads=n_heads, dropout=dropout)
            for i in range(n_layers)
        ])
        self.embed_dim = embed_dim

    def forward(self, obs: torch.Tensor):
        """obs: (B, 6 + max_humans * 6) relative-format observation."""
        node_feats, node_mask, robot_idx = build_graph_from_rel_obs(obs, self.max_humans)
        edge_feats = build_edge_feats(node_feats)

        x = self.node_embed(node_feats)
        for layer in self.layers:
            x = layer(x, edge_feats, node_mask)
            x = F.relu(x)

        return x[:, robot_idx, :]   # (B, embed_dim)


if __name__ == "__main__":
    B, max_humans = 4, 20
    obs_dim = 6 + max_humans * 6
    obs = torch.randn(B, obs_dim)
    blocks = obs[:, 6:].view(B, max_humans, 6)
    blocks[:, 3:, 5] = 0.0  # only first 3 humans visible in this batch

    enc = GraphAttentionEncoder(max_humans=max_humans, embed_dim=128)
    out = enc(obs)
    assert out.shape == (B, 128)
    print("gnn_encoder smoke test OK:", out.shape)