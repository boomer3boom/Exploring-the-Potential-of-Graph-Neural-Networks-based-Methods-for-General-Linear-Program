"""
This file test and collect the SIB Initial Basis Prediction.
See slack_path for SIB on slack.
See primal_[ath for SIB on primal
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree  # Importing required utilities
import wandb  # Import wandb
from arch import *

# Define the loss function (assuming binary classification)
criterion = nn.BCELoss()

graphs_path = "/home/ac/Learning_To_Pivot/bipartie_graphs2"
test_dataset = LPDataset(graphs_path, 0, 1000)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Load the model
input_dim = 2
hidden_dim = 64
model_slack = GCN(input_dim, hidden_dim)
model_primal = GCN(input_dim, hidden_dim)

slack_path = "/home/ac/Learning_To_Pivot/slack.pth"
primal_path = "/home/ac/Learning_To_Pivot/primal.pth"
slack_ml = torch.load(slack_path)
primal_ml = torch.load(primal_path)
model_slack.load_state_dict(slack_ml['model_state_dict'])
model_primal.load_state_dict(primal_ml['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_slack = model_slack.to(device)
model_primal = model_primal.to(device)
#model_parameters = model.state_dict()
#for param_name, param_tensor in model_parameters.items():
#    print(param_name, param_tensor.shape)
results_dir = "/home/ac/Learning_To_Pivot/SIB_result"
# Evaluation loop
model_slack.eval()
model_primal.eval()
with torch.no_grad():
    for idx, data in enumerate(test_loader):
        data = data.to(device)
        p_output = model_primal(data)
        s_output = model_slack(data)
        #print(output[1600:1900])
        collected_results = []
        index1 = 0
        index2 = 1200
        for _ in range(data.num_graphs):
            collected_p = p_output[index1:index1+500]
            collected_results.extend(collected_p.cpu().numpy())
            collected_s = s_output[index2:index2 + 700]
            collected_results.extend(collected_s.cpu().numpy())

        instance_folder = f"{results_dir}/SIB_{idx}"
        np.savez(instance_folder, 
            SIB = collected_results)
        print(f"Instance {idx}: Saved SIB results.")