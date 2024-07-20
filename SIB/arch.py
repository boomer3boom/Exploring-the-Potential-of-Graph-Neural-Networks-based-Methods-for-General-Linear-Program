"""
This file defines the Architecture for the GCN model on SIB.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree  # Importing required utilities
import wandb  # Import wandb

# Define GCNConv class
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops and calculate normalization
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=1.0, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return edge_attr.view(-1, 1) * x_j * norm.view(-1, 1)

    def update(self, aggr_out):
        # aggr_out is the aggregated message
        return self.lin(aggr_out)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        #self.conv3 = GCNConv(hidden_dim+32, hidden_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.lin = nn.Linear(hidden_dim, 1)  # Output dimension is 1 for binary classification

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #print(x[1500:1700])
        #x = torch.cat((x[:, :1], x[:, 2:]), dim=1)
        #print("x shape after removing the second feature:", x.shape)
        # Step 1: Message Passing from Variable to Constraint node
        x = F.tanh(self.conv1(x, edge_index, edge_attr))
        #x = self.dropout(x)
        #print(x[:10])
        # Step 2: Message Passing from Constraint to Variable node
        #x = self.prelu(self.conv2(x, edge_index, edge_attr))
        #print(x[:10])
        x = self.lin(x)  # Slice to get only the first 500 elements
        result = torch.sigmoid(x)
        return result  # Apply sigmoid to get probabilities (0 to 1)

class LPDataset(Dataset):
    def __init__(self, graphs_path, start, end):
        self.graphs_path = graphs_path
        self.graph_files = ["bipartite_graph_"+str(f)+".pt" for f in range(start, end)]
    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph_path = os.path.join(self.graphs_path, self.graph_files[idx])
        graph_data = torch.load(graph_path)
        
        # Ensure graph_data is an instance of torch_geometric.data.Data
        if not isinstance(graph_data, Data):
            raise TypeError('Expected graph_data to be an instance of torch_geometric.data.Data')

        return graph_data

def collate_fn(batch):
    batch = Batch.from_data_list(batch)

    return batch