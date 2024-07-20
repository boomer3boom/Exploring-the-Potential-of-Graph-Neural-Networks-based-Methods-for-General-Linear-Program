"""
This inferences SIB initial prediction to m variables. Where m is the number of constraints.
"""
import numpy as np
import os

lp_instance_folder = "/home/ac/Learning_To_Pivot/lp_instances2"
SIB_folder = "/home/ac/Learning_To_Pivot/SIB_result"

#Need to remove the padded constraints
def remove_padded_constraints(instance_num, lp_folder, SIB_folder):
    """
    Open the LP instance file and remove padding.
    """
    instance_folder = os.path.join(lp_folder, f"instance_{instance_num}.npz")
    SIB_instance_folder = os.path.join(SIB_folder, f"SIB_{instance_num}.npz")

    data = np.load(instance_folder)
    data2 = np.load(SIB_instance_folder)
    
    num_constraints = len(data['b'])
    result = data2['SIB']
    total = 1200 - (700 - num_constraints)
    new_result = result[:total]
    print(len(new_result))

    return new_result, num_constraints

def open_result(instance_num, lp_folder, SIB_folder):
    """
    Open the files
    """
    instance_folder = os.path.join(lp_folder, f"instance_{instance_num}.npz")
    SIB_instance_folder = os.path.join(SIB_folder, f"SIB_{instance_num}.npz")

    data = np.load(instance_folder)
    data2 = np.load(SIB_instance_folder)
    result = data2['SIB']
    return result    

def set_top_k_probabilities(probabilities, k):
    """
    Set the top k probability to be 1 and the rest 0.
    """
    # Get the top k indices
    top_k_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:k]
    
    # Initialize a result list with zeros
    result = [0] * len(probabilities)
    
    # Set top k probabilities to 1
    for idx in top_k_indices:
        result[idx] = 1

    return result

output_folder = "/home/ac/Learning_To_Pivot/SIB_basis"
for i in range(1000):
    #new_result, num_constraints = remove_padded_constraints(i,lp_instance_folder, SIB_folder)
    result = open_result(i,lp_instance_folder, SIB_folder)
    num_constraints = 700
    init_basis = set_top_k_probabilities(result, num_constraints)
    instance_folder = f"{output_folder}/basis_{i}"
    
    np.savez(instance_folder, 
        initial_basis = init_basis)
    print(f"Instance {i}: Saved basis results.")