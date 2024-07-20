"""
This transform the LP problem into bipartite graph for SIB model to train on.
"""
import numpy as np
import torch
import os

def scale_features(features, new_min, new_max):
    old_min = np.min(features)
    old_max = np.max(features)
    # Avoid division by zero
    if old_max - old_min == 0:
        return np.full(features.shape, new_min)
    scaled_features = (features - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_features

def load_lp_instance(instance_folder):
    data = np.load(instance_folder)
    c = data['c']
    A = data['A']
    b = data['b']
    optimal_solution = data['optimal_solution']
    return c, A, b, optimal_solution

def determine_max_sizes(num_instances, lp_folder):
    max_vars = 0
    max_constraints = 0
    
    for i in range(num_instances):
        instance_folder = os.path.join(lp_folder, f"instance_{i}.npz")
        data = np.load(instance_folder)
        num_vars = len(data['c']) - data['A'].shape[0]
        num_constraints = len(data['b'])
        
        if num_vars > max_vars:
            max_vars = num_vars
        if num_constraints > max_constraints:
            max_constraints = num_constraints
    
    return max_vars, max_constraints

import torch
from torch_geometric.data import Data
import numpy as np
import os

def pad_tensor(tensor, target_size, dim=0):
    pad_size = list(tensor.size())
    pad_size[dim] = target_size - pad_size[dim]
    padding = torch.zeros(*pad_size, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dim)

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def lp_to_bipartite_graph(c, A, b, optimal_solution, max_vars, max_constraints):
    num_vars = 1200
    num_constraints = len(b)
    
    # Calculate normalization parameters
    c_min, c_max = min(c), max(c)
    b_min, b_max = min(b), max(b)
    nnz_ratio_var_min, nnz_ratio_var_max = 0, 1
    cosine_sim_min, cosine_sim_max = -1, 1
    nnz_ratio_constraint_min, nnz_ratio_constraint_max = 0, 1
    
    # Variable node features
    variable_features = []
    dummy = []
    for i in range(num_vars):
        feature1_var = normalize(c[i], c_min, c_max)
        #print(feature1_var)
        #feature1_var = c[i]
        nnz_ratio_var = np.count_nonzero(A[:, i])/ num_constraints
        #nnz_ratio_var = normalize(np.count_nonzero(A[:, i]) / num_constraints, nnz_ratio_var_min, nnz_ratio_var_max)
        #if i < 500:
         #   dummy.append([feature1_var, nnz_ratio_var])
        variable_features.append([feature1_var, nnz_ratio_var])
        #variable_features.append([feature1_var])
    
    #dummy = np.array(dummy, dtype=np.float32)
    
    # Find min and max values of variable features
    # Find Q1 and Q3 of variable features
    #percentile_30 = np.percentile(dummy, 30, axis=0)
    #percentile_70 = np.percentile(dummy, 70, axis=0)
    variable_features = torch.tensor(variable_features, dtype=torch.float)
    
    # Constraint node features
    constraint_features = []
    for j in range(num_constraints):
        #constr_rhs = normalize(b[j], c_min, c_max)
        constr_rhs = normalize(b[j], b_min, b_max)
        #cosine_sim = np.dot(A[j], c) / (np.linalg.norm(A[j]) * np.linalg.norm(c))
        #cosine_sim = normalize(cosine_sim, cosine_sim_min, cosine_sim_max)
        nnz_ratio_constraint = np.count_nonzero(A[j])/ num_vars
        #nnz_ratio_constraint = normalize(np.count_nonzero(A[j]) / num_vars, nnz_ratio_constraint_min, nnz_ratio_constraint_max)
        constraint_features.append([constr_rhs, nnz_ratio_constraint])
        #constraint_features.append([constr_rhs])
    
    #constraint_features = np.array(constraint_features, dtype=np.float32)
    
    # Scale features to align with variable feature
    #constraint_features = scale_features(constraint_features, 0.3, 0.4)
    #print(constraint_features)
    #print(constraint_features)
    #print(constraint_features)
    constraint_features = torch.tensor(constraint_features, dtype=torch.float)
    
    # Padding node features
    variable_features = pad_tensor(variable_features, max_vars, dim=0)
    constraint_features = pad_tensor(constraint_features, max_constraints, dim=0)
    
    # Edges
    edge_index = []
    edge_attr = []
    
    for i in range(num_constraints):
        for j in range(num_vars):
            #if A[i][j] != 0:
            edge_index.append([j, num_vars + i])
            edge_attr.append(A[i][j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Labels
    labels = torch.tensor([1 if optimal_solution[i] > 1e-6 else 0 for i in range(1900)], dtype=torch.float)
    labels = pad_tensor(labels, 1900, dim=0)
    
    data = Data(x=torch.cat([variable_features, constraint_features], dim=0),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=labels)
    
    return data

def save_bipartite_graph(graph_data, instance_index, output_folder):
    torch.save(graph_data, os.path.join(output_folder, f"bipartite_graph_{instance_index}.pt"))
    print(f"Instance {instance_index}: Saved bipartite graph.")

def convert_all_lp_to_bipartite(num_instances, lp_folder, output_folder, max_vars, max_constraints):
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(num_instances):
        instance_folder = os.path.join(lp_folder, f"instance_{i}.npz")
        c, A, b, optimal_solution = load_lp_instance(instance_folder)
        graph_data = lp_to_bipartite_graph(c, A, b, optimal_solution, max_vars, max_constraints)
        save_bipartite_graph(graph_data, i, output_folder)

# Specify the folder containing the LP instances and the folder to save the bipartite graphs
lp_folder = "/home/ac/Learning_To_Pivot/lp_instances2"
bipartite_graph_folder = "/home/ac/Learning_To_Pivot/bipartie_graph2_replicate"
#bipartie_graphs2 -> normalised, nnz
#bipartie_graphs3 -> nomalised, none
#bipartie_graphs4 -> nomalised, none and remove 0 edges
#bipartie_graphs5 -> Unnormalised, none and remove 0 edges
# Determine the maximum sizes
max_vars, max_constraints = determine_max_sizes(1000, lp_folder)
print(f"Maximum number of variables: {max_vars}")
print(f"Maximum number of constraints: {max_constraints}")
# Convert all LP instances to bipartite graphs
convert_all_lp_to_bipartite(1000, lp_folder, bipartite_graph_folder, 1200, 700)