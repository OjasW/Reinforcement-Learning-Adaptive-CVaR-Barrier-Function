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

class BaseGNN(nn.Module):
    def __init__(self, max_humans: int, input_dim: int, hidden_dim: int, output_dim: int, num_layers=2):
        super(BaseGNN, self).__init__()
        self.max_humans = max_humans
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("BaseGNN is an abstract class. Please implement the forward method in a subclass.")

class GATModel(BaseGNN):
    def __init__(self, max_humans: int, input_dim: int, edge_dim: int, hidden_dim: int, output_dim: int, num_layers=2):
        super(GATModel, self).__init__(max_humans, input_dim, hidden_dim, output_dim, num_layers)

        # Define input dimension for layer z_ij
        # source and destination node features + edge features
        self.zij_dim = 2 * input_dim + edge_dim

        # MLP psi1: Encodes zij
        self.psi1 = nn.Sequential(nn.Linear(self.zij_dim, 256), nn.ReLU(), nn.Linear(256, 128))

        #MLP psi2: Computes attention weights
        self.psi2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

        # MLP psi3; Transforms edge specific messages
        self.psi3 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))

        # MLP psi4: Outputs final scalar
        self.psi4 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, data):
        """
        Forward pass for GNN

        Args:
            data: PyTorch Geometric data object containing node_features, edge_indices, edge_attributes, and batch

        Returns:
            Tensor: Predicted scalar values for each graph in the batch
        """

        node_features, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        node_valid = node_features[:, -1]

        # Step 1: Create z_ij = [v_i, v_j, e_ij] for each edge
        src, dest = edge_index
        v_i = node_features[src]
        v_j = node_features[dest]
        z_ij = torch.cat([v_i, v_j, edge_attr], dim=1)

        # Step 2: encode z_ij features via psi1
        q_ij = self.psi1(z_ij)

        # Step 3: Node wise softmax
        raw_weights = self.psi2(q_ij).squeeze(-1)
        dest_valid = node_valid[dest] > 0.5
        raw_weights = raw_weights.masked_fill(~dest_valid, float("-inf"))

        # scatter_softmax alternative
        max_weights = torch.full((raw_weights.size(0),), float("-inf"), device=raw_weights.device)
        max_weights.scatter_reduce_(0, src, raw_weights, reduce="amax")
        exp_weights = torch.exp(raw_weights - max_weights[src])
        sum_weights = torch.zeros_like(max_weights)
        sum_weights.scatter_add_(0, src, exp_weights)
        attention_weights = exp_weights / (sum_weights[src] + 1e-16)

        messages = self.psi3(q_ij)

        # Step 4: Aggregate messages to compute q_i for each node
        weighted_messages = attention_weights.unsqueeze(-1) * messages

        num_nodes = node_features.shape[0]

        # scatter_sum alternative
        q_i = torch.zeros(
            num_nodes,
            weighted_messages.size(1),
            device=weighted_messages.device,
            dtype=weighted_messages.dtype
        )
        q_i.index_add_(0, src, weighted_messages)

        # Step 5: Final regression via psi4
        num_graphs = int(batch.max().item()) + 1
        nodes_per_graph = self.max_humans + 2

        robot_indices = torch.arange(0, num_graphs * nodes_per_graph, nodes_per_graph, device=batch.device)
        robot_q = q_i[robot_indices]

        # Step 6: Pass the robot's feature vector through psi4
        output = self.psi4(robot_q)

        return output.squeeze(-1)
