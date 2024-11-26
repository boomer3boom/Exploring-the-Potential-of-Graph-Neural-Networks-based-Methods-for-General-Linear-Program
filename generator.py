"""
This file denotes how we generated our LP instances and ensure all LP instance is feasible.
"""

import numpy as np
from scipy.optimize import linprog

def generate_feasible_bounded_lp(m, n):
    # Randomly generate k, j, and h such that k + j + h <= m
    while True:
        first_pick = np.random.randint(200, 550)
        second_pick = np.random.randint(100, 300)
        third_pick = np.random.randint(150, 300)
        #k, j, h, remaing denotes how many <, >, <=, >= there should be in our constraints
        k = np.random.randint(0, first_pick)
        j = np.random.randint(0, int((m + 1 - k) / 3) + second_pick)
        h = np.random.randint(0, int((m + 1 - k - j)/2) + third_pick)
        remaining_constraints = m - k - j - h
        if remaining_constraints >= 0 and remaining_constraints + k + j + h == m:
            break

    # The code below outlines how small and large problems were generated with respect to inequalities.
    # rand_n = [ random.random() for i in range(4) ]
    # # extend the floats so the sum is approximately x (might be up to 3 less, because of flooring) 
    # result = [ math.floor(i * m / sum(rand_n)) for i in rand_n ] 
    # # randomly add missing numbers 
    # for i in range(m - sum(result)): 
    #     result[random.randint(0,3)] += 1

    # k = result[0]
    # j = result[1]
    # h = result[2]
    # remaining_constraints = result[3]
    
    print(f"Generated constraints: k = {k}, j = {j}, h = {h}, remaining = {remaining_constraints}")

    # Generate a feasible solution x
    x_feasible = np.random.rand(n)
    
    # Generate constraints of the form Ax < b
    #A_lt = np.random.rand(k, n)
    A_lt = np.random.uniform(-1, 1, size=(k, n))
    b_lt = A_lt @ x_feasible + np.random.rand(k)  # Ensure feasibility for Ax < b
    
    # Generate constraints of the form Ax > b
    #A_gt = np.random.rand(j, n)
    A_gt = np.random.uniform(-1, 1, size=(j, n))
    b_gt = A_gt @ x_feasible - np.random.rand(j)  # Ensure feasibility for Ax > b

    # Generate constraints of the form Ax <= b
    #A_leq = np.random.rand(h, n)
    A_leq = np.random.uniform(-1, 1, size=(h, n))
    b_leq = A_leq @ x_feasible + np.random.rand(h)  # Ensure feasibility for Ax <= b
    
    # Generate constraints of the form Ax >= b
    #A_geq = np.random.rand(remaining_constraints, n)
    A_geq = np.random.uniform(-1, 1, size=(remaining_constraints, n))
    b_geq = A_geq @ x_feasible - np.random.rand(remaining_constraints)  # Ensure feasibility for Ax >= b
    
    # Combine all constraints into a single matrix A and vector b
    A_combined = np.vstack([A_lt, A_leq, -A_gt, -A_geq])
    b_combined = np.hstack([b_lt, b_leq, -b_gt, -b_geq])

    # Shuffle A_combined and b_combined together
    combined = list(zip(A_combined, b_combined))
    np.random.shuffle(combined)
    A_combined, b_combined = zip(*combined)
    A_combined = np.array(A_combined)
    b_combined = np.array(b_combined)

    # Generate c with both positive and negative values
    c = np.random.uniform(-1, 1, size=n)
    
    # Add slack variables to convert inequalities to equalities
    num_constraints = A_combined.shape[0]
    A_slack = np.hstack((A_combined, np.eye(num_constraints)))
    c_slack = np.hstack((c, np.zeros(num_constraints)))


    #Due to the lack of 0s which is not practical, we include a method to adjust coefficients to 0
    tolerance = np.random.uniform(0, 0.3)
    random_values = np.random.rand(*A_slack.shape)  # Example random_values, adjust as needed

    coin_chance = np.random.uniform(0.1, 0.9)
    # Adjust coefficients close to zero
    mask = np.abs(A_slack) < tolerance
    random_mask = random_values > coin_chance
    A_slack[mask & random_mask] = 0.0

    return c_slack, A_slack, b_combined, x_feasible

def solve_and_save_lp_instance(instance_index, m, n, output_folder):
    # Generate LP problem instance
    f_b = False
    while f_b == False:
        c_slack, A_slack, b_combined, x_feasible = generate_feasible_bounded_lp(m, n)
    
        # Solve the LP problem using SciPy's linprog with the 'highs' method
        result = linprog(c_slack, A_ub=A_slack, b_ub=b_combined, method='highs')
        if result.success:
            #If LP is feasible then break otherwise continue till feasibility.
            f_b = True

    # Extract results
    optimal_value = result.fun
    print(optimal_value)

    tolerance = 1e-9
    in_basis = [1 if v > tolerance else 0 for v in result.x]
    slack_in_basis = [1 if s >= tolerance else 0 for s in result.slack]
    print(sum(slack_in_basis))
    print(sum(in_basis))

    in_basis = np.array(in_basis)
    slack_in_basis = np.array(slack_in_basis)
    optimal_basis = np.concatenate((in_basis, slack_in_basis))
    #basis_indices = np.where(optimal_basis == 1)[0].tolist()

    # Save LP problem and results to files
    instance_folder = f"{output_folder}/instance_{instance_index}"
    
    np.savez(instance_folder, 
             c=c_slack, A=A_slack, b=b_combined, 
             optimal_value=optimal_value, 
             optimal_solution=optimal_basis, 
             slack_variables=slack_in_basis)
    
    print(f"Instance {instance_index}: Saved LP problem and results.")

def find_average_inequalities(m, n, number):
    sum_k = 0
    sum_j = 0
    sum_h = 0
    sum_r = 0
    for n in range(number):
        while True:
            first_pick = np.random.randint(200, 550)
            second_pick = np.random.randint(100, 300)
            third_pick = np.random.randint(150, 300)
            #k, j, h, remaing denotes how many <, >, <=, >= there should be in our constraints
            k = np.random.randint(0, first_pick)
            j = np.random.randint(0, int((m + 1 - k) / 3) + second_pick)
            h = np.random.randint(0, int((m + 1 - k - j)/2) + third_pick)
            remaining_constraints = m - k - j - h
            if remaining_constraints >= 0 and remaining_constraints + k + j + h == m:
                break
        
        sum_k += k
        sum_j += j
        sum_h += h
        sum_r += remaining_constraints
    
    return sum_k/number, sum_j/number, sum_h/number, sum_r/number
    

m_min = 600 
m_max = 700
n_min = 400
n_max = 500
num_instances = 1000
output_folder = "/home/ac/Learning_To_Pivot/lp_instances2"  # folder to save instances

import os
os.makedirs(output_folder, exist_ok=True)

for i in range(num_instances):
    # Randomly choose m between m_min and m_max
    #m = np.random.randint(m_min, m_max + 1)
    m = 700
    #n = np.random.randint(n_min, n_max + 1)
    n = 500
    
    # Solve and save instance
    solve_and_save_lp_instance(i, m, n, output_folder)
