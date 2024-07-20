"""
This takes the Experts choice that we collected, and transfor the LP instance and expert choice 
into a Bipartite graph for the LTP model.
"""

import numpy as np
import torch
import os
from torch_geometric.data import Data

def load_lp_instance(instance_folder):
    data = np.load(instance_folder)
    c = data['c']
    A = data['A']
    b = data['b']
    return c, A, b

def load_pivot_steps(instance_folder):
    pivots = np.load(instance_folder)
    return pivots

def load_initial_basis(instance_folder):
    ml_result = np.load(instance_folder)
    init_basis = ml_result['initial_basis']
    return init_basis

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def save_bipartite_graph(graph_data, instance_index, output_folder, pivot_step):
    torch.save(graph_data, os.path.join(output_folder, f"bipartite_graph_{instance_index}_{pivot_step}.pt"))
    print(f"Instance {instance_index}: Saved bipartite graph pivot {pivot_step}.")

def lp_to_bipartite_graph(c, A, b, pivots, init_basis, instance_index, output_folder):
    num_vars = 1200
    num_constraints = len(b)
    
    # Calculate normalization parameters
    c_min, c_max = min(c), max(c)
    b_min, b_max = min(b), max(b)

    variable_basis = init_basis[:500]
    slack_basis = init_basis[500:]
    zeros = np.zeros(700, dtype=int)
    cur_basis = np.concatenate((variable_basis, zeros, slack_basis))
    
    for step in range(len(pivots) + 1):
        
        # Variables
        variable_features = []
        for i in range(num_vars):
            feature1_var = normalize(c[i], c_min, c_max)
            nnz_ratio_var = np.count_nonzero(A[:, i]) / num_constraints
            in_basis = cur_basis[i]
            variable_features.append([feature1_var, nnz_ratio_var, in_basis])
        
        variable_features = torch.tensor(variable_features, dtype=torch.float)
        
        # Constraints
        constraint_features = []
        for j in range(num_constraints):
            constr_rhs = normalize(b[j], b_min, b_max)
            nnz_ratio_constraint = np.count_nonzero(A[j]) / num_vars
            in_basis = cur_basis[1200 + j]
            constraint_features.append([constr_rhs, nnz_ratio_constraint, in_basis])

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

        # Labels [Entering_Variable, Leaving Variable]
        labels = np.zeros((1900, 2))
        if step < len(pivots):
            entering = pivots[step][0]
            leaving = pivots[step][1]

            if entering >= 500:
                entering += 700
            if leaving >= 500:
                leaving += 700
    
            labels[entering][0] = 1
            labels[leaving][1] = 1

        labels = torch.tensor(labels, dtype=torch.float)

        # Is_optimal: check if current basis is optimal
        is_optimal = torch.tensor([step == len(pivots)], dtype=torch.float)

        # Save the data
        data = Data(x=torch.cat([variable_features, constraint_features], dim=0),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=labels,
                    is_optimal=is_optimal)
        
        save_bipartite_graph(data, instance_index, output_folder, step)

        # Update cur_basis
        if step < len(pivots):
            cur_basis[leaving] = 0
            cur_basis[entering] = 1

    return

def convert_all_lp_to_bipartite(num_instances, lp_folder, output_folder, pivot_folder, init_basis_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(900, num_instances):
        lp_instance_folder = os.path.join(lp_folder, f"instance_{i}.npz")
        pivot_instance_folder = os.path.join(pivot_folder, f"pivoting_steps_{i}.npy")
        init_basis_instance_folder = os.path.join(init_basis_folder, f"basis_{i}.npz")
        c, A, b = load_lp_instance(lp_instance_folder)
        pivots = load_pivot_steps(pivot_instance_folder)
        init_basis = load_initial_basis(init_basis_instance_folder)
        lp_to_bipartite_graph(c, A, b, pivots, init_basis, i, output_folder)

# Specify the folder containing the LP instances and the folder to save the bipartite graphs
lp_folder = "/home/ac/Learning_To_Pivot/lp_instances2"
pivot_folder = "/home/ac/Learning_To_Pivot/pivoting_steps"
basis_folder = "/home/ac/Learning_To_Pivot/SIB_basis"
bipartite_graph_folder = "/home/ac/Learning_To_Pivot/pivot_test"

# Convert all LP instances to bipartite graphs
#convert_all_lp_to_bipartite(1000, lp_folder, bipartite_graph_folder, pivot_folder, basis_folder)