import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

def create_graph(obstacles: torch.Tensor, max_humans: int) -> Dict:
    """
    Create a graph from the observation tensor.

    Args:
        obs: (B, 6 + max_humans * 6) relative-format observation.
        max_humans: Maximum number of humans in the environment.
        B = batch size
        max_humans = maximum number of humans in the environment
    Returns:
        graph_data: A dictionary containing node features, edge indices, and edge attributes.
    """
    """
    Same node ordering/topology as create_graph() (robot=0, obstacles=1..K,
    goal=K+1) and the same batching machinery, but with the node/edge split
    inverted:
 
    Node features (4 dims): [is_robot, is_obstacle, is_goal, mask]
        Purely categorical/structural. No position, velocity, radius, or
        heading lives on a node anymore.
 
    Edge features (7 dims): [rel_x, rel_y, rel_vx, rel_vy, surface_dist,
                              cos(bearing_offset), sin(bearing_offset)]
        - rel_x/y, rel_vx/y: identical construction to create_graph's
          rel_positions/rel_velocities.
        - surface_dist: center-to-center distance minus the sum of the two
          nodes' radii, clamped at 0 -> the GNN sees clearance, not raw
          distance, without radius ever needing to be a node feature.
        - bearing_offset: angle between the robot's heading and the
          direction to node j, i.e. "is this obstacle ahead / to my left /
          to my right". Only nonzero on edges out of the robot node (index
          0) since no other node has a heading; zero everywhere else.
 
    All positions/velocities/the -goal_rel_pos sign convention are reused
    verbatim from create_graph()'s construction, so this is guaranteed to
    encode the same relations as the already-validated version -- only
    which tensor (node vs. edge) each quantity lives on has changed.
    """
    B, D = obstacles.shape
    expected_dim = 6 + 6 * max_humans
    if D != expected_dim:
        raise ValueError(
            f"Expected observation dimension {expected_dim} "
            f"for max_humans={max_humans}, got {D}"
        )
 
    device = obstacles.device
    K = max_humans
    num_nodes = K + 2
    robot_idx, goal_idx = 0, K + 1
 
    goal_rel_pos  = obstacles[:, :2]     # (B, 2)
    robot_vel     = obstacles[:, 2:4]    # (B, 2)
    robot_heading = obstacles[:, 4:5]    # (B, 1)
    robot_radius  = obstacles[:, 5:6]    # (B, 1)
 
    blocks = obstacles[:, 6:].reshape(B, K, 6)
    obs_rel_pos = blocks[:, :, 0:2]   # (B, K, 2)
    obs_vel     = blocks[:, :, 2:4]   # (B, K, 2)
    obs_radius  = blocks[:, :, 4:5]   # (B, K, 1)
    obs_mask    = blocks[:, :, 5:6]   # (B, K, 1)
 
    # Per-node position/velocity/radius/mask, same convention as create_graph
    # (robot at the origin of its own relative frame; goal uses the same
    # -goal_rel_pos sign flip create_graph already uses).
    all_positions = torch.cat([
        torch.zeros(B, 1, 2, device=device),
        obs_rel_pos,
        -goal_rel_pos.unsqueeze(1),
    ], dim=1)  # (B, N, 2)
 
    all_velocities = torch.cat([
        robot_vel.unsqueeze(1),
        obs_vel,
        torch.zeros(B, 1, 2, device=device),
    ], dim=1)  # (B, N, 2)
 
    all_radii = torch.cat([
        robot_radius,
        obs_radius.squeeze(-1),
        torch.zeros(B, 1, device=device),
    ], dim=1)  # (B, N)
 
    all_mask = torch.cat([
        torch.ones(B, 1, device=device),
        obs_mask.squeeze(-1),
        torch.ones(B, 1, device=device),
    ], dim=1)  # (B, N)
 
    # ---- node features: one-hot type + mask only ----
    type_onehot = torch.zeros(B, num_nodes, 3, device=device)
    type_onehot[:, robot_idx, 0] = 1.0
    type_onehot[:, 1:K + 1, 1] = 1.0
    type_onehot[:, goal_idx, 2] = 1.0
    node_features = torch.cat([type_onehot, all_mask.unsqueeze(-1)], dim=-1)  # (B, N, 4)
 
    # ---- pairwise quantities -> edges ----
    rel_positions  = all_positions.unsqueeze(2) - all_positions.unsqueeze(1)    # (B, N, N, 2)
    rel_velocities = all_velocities.unsqueeze(2) - all_velocities.unsqueeze(1)  # (B, N, N, 2)
    center_dist    = torch.norm(rel_positions, dim=-1)                          # (B, N, N)
    radii_sum      = all_radii.unsqueeze(2) + all_radii.unsqueeze(1)            # (B, N, N)
    surface_dist   = torch.clamp(center_dist - radii_sum, min=0.0).unsqueeze(-1)  # (B, N, N, 1)
 
    # bearing_to_j computed directly from all_positions (robot is the origin
    # of this frame, so all_positions[:, j] already points from robot to j).
    # atan2(0, 0) safely returns 0 in torch, so the undefined j=robot_idx
    # entry never nans out -- and it's sliced away anyway since edge_index
    # excludes self-loops.
    bearing_to_j    = torch.atan2(all_positions[:, :, 1], all_positions[:, :, 0])  # (B, N)
    bearing_offset  = bearing_to_j - robot_heading                                 # (B, N)
    bearing_feat = torch.zeros(B, num_nodes, num_nodes, 2, device=device)
    bearing_feat[:, robot_idx, :, 0] = torch.cos(bearing_offset)
    bearing_feat[:, robot_idx, :, 1] = torch.sin(bearing_offset)
 
    edge_features = torch.cat(
        [rel_positions, rel_velocities, surface_dist, bearing_feat], dim=-1
    )  # (B, N, N, 7)
 
    # ---- assemble into a PyG batch (identical machinery to create_graph) ----
    batch = torch.arange(B, device=device).repeat_interleave(num_nodes)
 
    edge_index = torch.tensor(
        [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j],
        dtype=torch.long, device=device,
    ).t().contiguous()
    E = edge_index.shape[1]
 
    batch_offsets = (torch.arange(B, device=device) * num_nodes).view(B, 1, 1)
    batched_edge_index = (edge_index.unsqueeze(0) + batch_offsets).permute(1, 0, 2).reshape(2, B * E)
 
    edge_attr = edge_features[:, edge_index[0], edge_index[1]].reshape(B * E, 7)
 
    return Data(
        x=node_features.reshape(B * num_nodes, 4),
        edge_index=batched_edge_index,
        edge_attr=edge_attr.view(-1, 7),
        batch=batch,
    )

def _sanity_check_relational_graph():
    """
    Hand-constructed check, same style as the rest of this project's
    validation: build a graph from a batch of 2 samples with a known
    obstacle geometry and assert specific traceable values, not just shapes.
    Run this directly (python graph.py --check) before wiring the new
    encoder into diff_cvar.py's forward pass.
    """
    torch.manual_seed(0)
    B, max_humans = 2, 3
 
    obs = torch.zeros(B, 6 + 6 * max_humans)
    obs[:, :2] = torch.tensor([[3.0, 0.0], [0.0, 0.0]])   # goal_rel_pos
    obs[:, 2:4] = torch.tensor([[1.0, 0.0], [0.0, 0.0]])  # robot_vel
    obs[:, 4] = torch.tensor([0.0, 0.0])                  # robot_heading (facing +x)
    obs[:, 5] = 0.3                                        # robot_radius
 
    # sample 0: one real obstacle directly ahead of the robot, two padded slots
    blocks = obs[0, 6:].reshape(max_humans, 6)
    blocks[0] = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.2, 1.0])  # ahead, real
    blocks[1] = torch.zeros(6)                                 # masked out
    blocks[2] = torch.zeros(6)                                 # masked out
    obs[0, 6:] = blocks.reshape(-1)
 
    data = create_graph(obs, max_humans)
 
    num_nodes = max_humans + 2
    assert data.x.shape == (B * num_nodes, 4), data.x.shape
    assert data.edge_attr.shape[-1] == 7, data.edge_attr.shape
 
    # node one-hots: robot=idx0, obstacles=idx1..3, goal=idx4, per sample
    robot_row = data.x[0]
    assert torch.equal(robot_row, torch.tensor([1., 0., 0., 1.])), robot_row
    real_obs_row = data.x[1]
    assert torch.equal(real_obs_row, torch.tensor([0., 1., 0., 1.])), real_obs_row
    masked_obs_row = data.x[2]
    assert torch.equal(masked_obs_row, torch.tensor([0., 1., 0., 0.])), masked_obs_row  # mask=0
 
    # edge (robot -> real obstacle): obstacle is straight ahead (heading=0),
    # so bearing_offset should be ~0 -> cos~=1, sin~=0
    edge_index = data.edge_index[:, :12]  # first sample's block of edges
    src, dst = data.edge_index
    mask_e = (src == 0) & (dst == 1)
    edge = data.edge_attr[mask_e][0]
    assert torch.isclose(edge[4], torch.tensor(1.5), atol=1e-4), f"surface_dist expected 2.0-(0.3+0.2)=1.5, got {edge[4]}"
    assert torch.isclose(edge[5], torch.tensor(1.0), atol=1e-4), f"cos(bearing) expected ~1.0, got {edge[5]}"
    assert torch.isclose(edge[6], torch.tensor(0.0), atol=1e-4), f"sin(bearing) expected ~0.0, got {edge[6]}"
 
    print("[PASS] _sanity_check_relational_graph: node one-hots, mask, "
          "surface_dist, and bearing_offset all match hand-computed values")


def visualize_graph(graph_data: Data, max_humans: int):
    """
    Visualize a PyTorch Geometric graph containing:
        node 0              : robot
        nodes 1..K          : obstacles
        node K+1            : goal

    Edge attributes:
        [relative_x, relative_y,
         relative_vx, relative_vy,
         distance]
    """
    
    # Convert PyG graph to NetworkX
    # Create NetworkX graph directly from edge_index
    nx_graph = nx.DiGraph()

    nx_graph.add_nodes_from(range(graph_data.num_nodes))

    edges = graph_data.edge_index.detach().cpu().numpy().T
    nx_graph.add_edges_from(edges)

    # Extract physical node positions from node features
    # node_features = graph_data.x.detach().cpu()
    # node_positions = node_features[:, :2].numpy()

    # Create Networkx positions
    pos = nx.spring_layout(nx_graph, seed=42)

    # Plot
    plt.figure(figsize=(10, 10))

    # Edges
    nx.draw_networkx_edges(nx_graph, pos, arrows=False, alpha=0.4)

     # Node groups
    robot = [0]
    obstacles = list(range(1, max_humans + 1))
    goal = [max_humans + 1]

    # Draw robot
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=robot, node_size=700, node_color="red", node_shape="o")
    # Draw obstacles
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=obstacles, node_size=500, node_color="lightblue", node_shape="o")
    # Draw goal
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=goal, node_size=700, node_color="green", node_shape="o")
    
    # Node labels
    labels = {0: "R"}
    for i in obstacles:
        labels[i] = f"O{i}"
    labels[max_humans + 1] = "G"

    nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=10)

    # edge_labels = nx.get_edge_attributes(nx_graph, "edge_attr")
    # edge_labels = {edge: f"{float(attr[4]):.2f}" for edge, attr in edge_labels.items()}

    # nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=7)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("GNN Graph")
    plt.axis("equal")
    plt.grid(False)
    plt.show()

def main():

    max_humans = 20
    num_nodes = max_humans + 2

    # Create graph topology only
    edge_index = torch.tensor(
        [
            (i, j)
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j
        ],
        dtype=torch.long
    ).t().contiguous()

    # Placeholder node/edge features.
    # These are NOT environment data; they only satisfy the
    # PyG Data structure required by the visualization.
    node_features = torch.zeros(num_nodes, 8)
    edge_attr = torch.zeros(edge_index.shape[1], 5)

    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    print(graph_data)
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of edges: {graph_data.num_edges}")

    visualize_graph(graph_data, max_humans)


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        _sanity_check_relational_graph()
    else:
        main()
