"""
This test the entire framework of 2 SIB + LTP.
It does so in these steps:
#Step 1 get init_basis
#Step 2 transform the instance into bipartite graph
#Step 3 run the pivot_expert
#Step 4 update the graph with
#Step 5 back to step 3 until condition met for terminattion.
# """

from bipatite_pivot_transform import load_lp_instance, load_initial_basis, normalize
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from pivot_arch import *
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import os
from scipy.optimize import linprog
import numpy as np
import pandas as pd

#Return the uniform detail (just the features and edge)
def bipartite_graph_details(c, A, b):
    """
    Gets the LP instance detail for a bipartite graph trnsformation.
    But do not turn it into bipartite graph. It returns the variable 
    and constraint features as well as the edges.
    """
    num_vars = 1200
    num_constraints = len(b)

    # Calculate normalization parameters
    c_min, c_max = min(c), max(c)
    b_min, b_max = min(b), max(b)

    # Variables
    variable_features = []
    for i in range(num_vars):
        feature1_var = normalize(c[i], c_min, c_max)
        nnz_ratio_var = np.count_nonzero(A[:, i]) / num_constraints
        variable_features.append([feature1_var, nnz_ratio_var])
    
    variable_features = torch.tensor(variable_features, dtype=torch.float)
    
    # Constraints
    constraint_features = []
    for j in range(num_constraints):
        constr_rhs = normalize(b[j], b_min, b_max)
        nnz_ratio_constraint = np.count_nonzero(A[j]) / num_vars
        constraint_features.append([constr_rhs, nnz_ratio_constraint])
    
    constraint_features = torch.tensor(constraint_features, dtype=torch.float)

    # Edges
    edge_index = []
    edge_attr = []

    for i in range(num_constraints):
        for j in range(num_vars):
            edge_index.append([j, num_vars + i])
            edge_attr.append(A[i][j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return variable_features, constraint_features, edge_index, edge_attr

def get_bipartite(variable_features, constraint_features, edge_index, edge_attr, cur_basis):
    """
    Turn the problem into a bipartite graph. It will also need to add what the current basis
    is as a feature for variable and constraint nodes.
    """
    num_vars = 1200
    num_constraints = 700
    
    variable_features = torch.cat((variable_features, torch.tensor(cur_basis[:num_vars], dtype=torch.float).view(-1, 1)), dim=1)
    constraint_features = torch.cat((constraint_features, torch.tensor(cur_basis[num_vars:], dtype=torch.float).view(-1, 1)), dim=1)

    data = Data(x=torch.cat([variable_features, constraint_features], dim=0),
                edge_index=edge_index,
                edge_attr=edge_attr)
    return data

basis_folder = "/home/ac/Learning_To_Pivot/SIB_basis"
lp_folder = "/home/ac/Learning_To_Pivot/lp_instances2"
result_folder = "Learning_To_Pivot/final_results"

torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PivotGCN(input_dim=3, hidden_dim=128, output_dim=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()
checkpoint_path = "/home/ac/Learning_To_Pivot/checkpoint2.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
else:
    start_epoch = 0
    best_loss = float('inf')

print(start_epoch)
model.eval()
results = []
for instance in range(801, 1000):
    init_basis_instance_folder = os.path.join(basis_folder, f"basis_{instance}.npz")
    init_basis = load_initial_basis(init_basis_instance_folder)
    
    variable_basis = init_basis[:500]
    slack_basis = init_basis[500:]
    zeros = np.zeros(700, dtype=int)
    cur_basis = np.concatenate((variable_basis, zeros, slack_basis))

    lp_instance_folder = os.path.join(lp_folder, f"instance_{instance}.npz")
    data = np.load(lp_instance_folder)
    c = data['c']
    A = data['A']
    b = data['b']
    optimal_basis = data['optimal_solution']

    # Get the data needed
    variable_features, constraint_features, edge_index, edge_attr = bipartite_graph_details(c, A, b)

    # Initialize a counter for pivots
    num_pivots = 0
    repeat_counter = {}

    matching_ones_before_pivot = 0
    for i in range(len(cur_basis)):
        if cur_basis[i] == 1 and optimal_basis[i] == 1:
            matching_ones_before_pivot += 1
    
    ratio_before_pivot = matching_ones_before_pivot / 700

    for steps in range(300):
        data = get_bipartite(variable_features, constraint_features, edge_index, edge_attr, cur_basis)
        data = data.to(device)
        out, optimal = model(data.x, data.edge_index, data.edge_attr, data.batch)
        feature_0 = out[:, 0]
        copy_0 = feature_0.detach().clone()
        feature_1 = out[:, 1]
        copy_1 = feature_1.detach().clone()

        # Select the entering variable
        for j in range(3000):
            entering = torch.argmax(copy_0-feature_1).item()
            
            if entering >= 500 and entering <= 1200:
                copy_0[entering] = float('-inf')
                continue
            
            if cur_basis[entering] == 1:
                copy_0[entering] = float('-inf')
                continue
            else:
                if steps == 0: 
                    entry_diff = feature_0[entering]-feature_1[entering]
                elif (feature_0[entering]-feature_1[entering])/entry_diff < 0.4:
                    j = 3000
                
                break

        if j == 3000:
            print("No suitable entering variables")
            break
        
        # Select the leaving variable
        for j in range(3000):
            leaving = torch.argmax(copy_1-feature_0).item()
            
            if leaving >= 500 and leaving <= 1200:
                copy_1[leaving] = float('-inf')
                continue
            
            if cur_basis[leaving] == 0:
                copy_1[leaving] = float('-inf')
                continue
            else:
                if steps == 0: 
                    leave_diff = feature_1[leaving]-feature_0[leaving]
                elif (feature_1[leaving]-feature_0[leaving])/leave_diff < 0.4:
                    j = 3000
                
                print(out[leaving])
                break
        
        if j == 3000:
            print("No suitable leaving variables")
            break

        current_step = (entering, leaving)

        if current_step in repeat_counter:
            repeat_counter[current_step] += 1
        else:
            repeat_counter[current_step] = 1
        
        if repeat_counter[current_step] > 4:
            break
        elif entering == leaving:
            break
        
        cur_basis[leaving] = 0
        cur_basis[entering] = 1
        num_pivots += 1
    
    matching_ones_after_pivot = 0
    for i in range(len(cur_basis)):
        if cur_basis[i] == 1 and optimal_basis[i] == 1:
            matching_ones_after_pivot += 1
    
    ratio_after_pivot = matching_ones_after_pivot / 700
    # Append data to list
    results.append({
        'Instance': instance,
        'SIB Ratio': ratio_before_pivot,
        'SIB+Pivot Ratio': ratio_after_pivot,
        'No.Pivot': num_pivots
    })
    print(f"{instance} done")

# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(results)

# Save ratios to a CSV file
df.to_csv(os.path.join(result_folder, "ratios_data_with_take.csv"), index=False)