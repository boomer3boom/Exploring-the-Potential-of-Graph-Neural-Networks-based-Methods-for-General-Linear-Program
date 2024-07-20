"""
Contains the LTP architecture of GCN and LPDDataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset, Batch
import os
import re
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree  # Importing required utilities

class GCNConv(MessagePassing):
    """
    Adjust the GCNConv slightly mainly the self-loop edge.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Adjust the GCNConv slightly mainly the self-loop edge. 
        Also normalise these edge attributes.
        """
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

class PivotGCN(nn.Module):
    """
    Specify our PivotGCN Architecture
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(PivotGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        #self.fc1 = nn.ReLU(hidden_dim, hidden_dim)
        #self.fc1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc_optimal = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index, edge_attr, batch):
        for count, conv in enumerate(self.convs):
            x = F.tanh(conv(x, edge_index, edge_attr))

        out = F.sigmoid(self.fc2(x))

        graph_embedding = global_mean_pool(x, batch)
        optimal = torch.sigmoid(self.fc_optimal(graph_embedding))
        
        return out, optimal

class LPDataset(Dataset):
    def __init__(self, graphs_path, end=8000):
        self.graphs_path = graphs_path
        self.graph_files = [f for count, f in enumerate(os.listdir(graphs_path)) if f.startswith("bipartite_graph_") and f.endswith(".pt") and count < end]
    
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