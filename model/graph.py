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
    B, D = obstacles.shape
    expected_dim = 6 + 6 * max_humans

    if D != expected_dim:
        raise ValueError(
            f"Expected observation dimension {expected_dim} "
            f"for max_humans={max_humans}, got {D}"
        )

    # Nodes are robot, goal and obstacles. Features are relative positions, velocities, radius, and validity mask.

    # Position (obstacle-major layout: each obstacle occupies a contiguous
    # 6-value block [rel_x, rel_y, vx, vy, radius, mask], NOT grouped by field)
    goal_rel_pos = obstacles[:, :2]  # (B, 2)

    blocks = obstacles[:, 6:].reshape(B, max_humans, 6)  # (B, max_humans, 6)
    obs_rel_pos = blocks[:, :, 0:2]   # (B, max_humans, 2)

    # Velocity
    robot_vel = obstacles[:, 2:4]  # (B, 2)
    obs_vel = blocks[:, :, 2:4]       # (B, max_humans, 2)

    # Radius
    robot_radius = obstacles[:, 5:6]   # (B, 1)
    obs_radius = blocks[:, :, 4:5]    # (B, max_humans, 1)

    # Heading angle
    robot_heading = obstacles[:, 4:5]  # (B, 1)

    # Obstacle mask
    obs_mask = blocks[:, :, 5:6]      # (B, max_humans, 1)

    # Node features = [relative position x, relative position y, velocity x, velocity y, radius, cos(heading angle), sin(heading angle), validity mask]
    robot_node = torch.stack([
        torch.zeros(B, device=obstacles.device),
        torch.zeros(B, device=obstacles.device),
        robot_vel[:, 0],
        robot_vel[:, 1],
        robot_radius[:, 0],
        torch.cos(robot_heading[:, 0]),
        torch.sin(robot_heading[:, 0]),
        torch.ones(B, device=obstacles.device),
    ], dim=-1)  # (B, 8)
    
    goal_node = torch.stack([
        -goal_rel_pos[:, 0],
        -goal_rel_pos[:, 1],
        torch.zeros(B, device=obstacles.device),
        torch.zeros(B, device=obstacles.device),
        torch.zeros(B, device=obstacles.device),
        torch.zeros(B, device=obstacles.device),
        torch.zeros(B, device=obstacles.device),
        torch.ones(B, device=obstacles.device),
    ], dim=-1)  # (B, 8)

    obstacle_nodes = torch.cat([
        obs_rel_pos,
        obs_vel,
        obs_radius,
        torch.zeros(B, max_humans, 2, device=obstacles.device),
        obs_mask,
    ], dim=-1)  # (B, K, 8)


    node_features = torch.cat([
        robot_node.unsqueeze(1),
        obstacle_nodes,
        goal_node.unsqueeze(1),
    ], dim=1)  # (B, K+2, 8)

    B, num_nodes, _ = node_features.shape
    batch = torch.arange(B, device=node_features.device).repeat_interleave(num_nodes)

    # Edge features = [relative position x, relative position y, relative velocity x, relative velocity y, relative distance]
    all_positions = torch.cat([robot_node[:, :2].unsqueeze(1), obs_rel_pos, goal_node[:, :2].unsqueeze(1)], dim=1)  # (B, K+2, 2)
    all_velocities = torch.cat([robot_node[:, 2:4].unsqueeze(1), obs_vel, goal_node[:, 2:4].unsqueeze(1)], dim=1)  # (B, K+2, 2)

    rel_positions = all_positions.unsqueeze(2) - all_positions.unsqueeze(1)  # (B, K+2, K+2, 2)
    rel_velocities = all_velocities.unsqueeze(2) - all_velocities.unsqueeze(1)  # (B, K+2, K+2, 2)
    rel_distances = torch.norm(rel_positions, dim=-1, keepdim=True)  # (B, K+2, K+2, 1)

    edge_features = torch.cat([rel_positions, rel_velocities, rel_distances], dim=-1)  # (B, K+2, K+2, 5)

    # Create edge indices and edge attributes
    edge_index = torch.tensor([(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long, device=node_features.device).t().contiguous()  # (2, E)

    E = edge_index.shape[1]

    batch_offsets = (torch.arange(B, device=node_features.device) * num_nodes).view(B, 1, 1)
    batched_edge_index = edge_index.unsqueeze(0) + batch_offsets
    batched_edge_index = batched_edge_index.permute(1, 0, 2).reshape(2, B * E)

    edge_attr = edge_features[:, edge_index[0], edge_index[1]].reshape(B * E, 5)  # (B * E, 5)

    return Data(
        x=node_features.reshape(B * num_nodes, 8),  # (B*(K+2), 8)
        edge_index=batched_edge_index,  # (2, B*E)
        edge_attr=edge_attr.view(-1, 5),  # (B*E, 5)
        batch=batch 
       )

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
    main()