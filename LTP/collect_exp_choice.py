"""
This file is responsible for saving the Pivoting experts choice as an NP file.
"""
import numpy as np
import os
from scipy.optimize import linprog
from expert import Expert1

lp_folder = "/home/ac/Learning_To_Pivot/lp_instances2/"
basis_folder = "/home/ac/Learning_To_Pivot/SIB_basis/"
output_folder = "/home/ac/Learning_To_Pivot/pivoting_steps/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

for idx in range(1000):
    # Load LP problem instance data
    instance_file = os.path.join(lp_folder, f"instance_{idx}.npz")
    init_basis_file = os.path.join(basis_folder, f"basis_{idx}.npz")

    data = np.load(instance_file)
    c = data['c']
    A = data['A']
    b = data['b']
    optimal_basis = data['optimal_solution']

    part1 = optimal_basis[:500]
    part2 = optimal_basis[1200:]

    optimal_basis = np.concatenate((part1,part2))

    ml_result = np.load(init_basis_file)
    init_basis = ml_result['initial_basis']

    # Indices for initial and optimal basis
    opt_basis_indices = np.where(optimal_basis == 1)[0].tolist()
    if len(opt_basis_indices) != 700:
        print(f"LP Instance {idx} is Faulty: Invalid number of variables")
        break
    
    cur_basis_indices = np.where(init_basis == 1)[0].tolist()

    # Instantiate Expert1
    expert = Expert1(c, A, b, opt_basis_indices, cur_basis_indices)

    # Collect expert's pivoting steps
    pivoting_steps = []
    repeat_counter = {}
    repetition = False
    while True:
        entering_var, leaving_var = expert.simplex_step()

        if entering_var == -1 or leaving_var == -1:
            break
        
        current_step = (entering_var, leaving_var)

        # Check if current step has been seen before
        if current_step in pivoting_steps:
            # Increase repeat count for this step
            if current_step in repeat_counter:
                repeat_counter[current_step] += 1
            else:
                repeat_counter[current_step] = 1
            
            # Check if the number of repeats exceeds 5
            if repeat_counter[current_step] > 5:
                print(f"Repetition detected in instance {idx}")
                repetition = True
                break
        
        # Add current step to pivoting_steps
        pivoting_steps.append(current_step)

    if repetition == True:
        break
    
    pivoting_steps = np.array(pivoting_steps)

    # Save as a .npy file in the output folder
    output_file = os.path.join(output_folder, f"pivoting_steps_{idx}.npy")
    np.save(output_file, pivoting_steps)

    print(f"Saved pivoting steps for instance {idx} to {output_file}")