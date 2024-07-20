"""
This defines the Pivoting expert class.
"""
import numpy as np
from scipy.optimize import linprog

class Expert1:
    """
    Defines the Pivoting expert class. The expert makes pivot and shows us its choice of
    entering and leaving variables.
    """
    def __init__(self, c, A, b, optimal_basis, current_basis):
        """
        Initialize the Expert1 class.

        Args:
        c : np.array, shape (n,)
            Cost coefficients.
        A : np.array, shape (m, n)
            Constraint matrix.
        b : np.array, shape (m,)
            Right-hand side vector.
        optimal_basis : list of int
            Indices of variables in the optimal basis.
        current_basis : list of int
            Indices of variables in the current basis.
        """
        self.c = c
        self.A = A
        self.b = b
        self.optimal_basis = optimal_basis
        self.current_basis = current_basis
        self.m, self.n = A.shape

    def steepest_edge_scores(self):
        """
        Compute the steepest edge scores for all non-basic variables.

        Returns:
        scores : np.array, shape (n,)
            Steepest edge scores for all variables.
        """
        B_inv = np.linalg.inv(self.A[:, self.current_basis])
        scores = np.zeros(self.n)
        for j in range(self.n):
            if j not in self.current_basis:
                aj_hat = B_inv @ self.A[:, j]
                scores[j] = self.c[j] / np.linalg.norm(aj_hat)
        return scores

    def select_entering_variable(self):
        """
        Select the entering variable based on the optimal basis and steepest edge scores.

        Returns:
        int
            Index of the entering variable.
        """
        scores = self.steepest_edge_scores()
        entering_variable = -1
        best_score = float('-inf')
        for j in self.optimal_basis:
            if j not in self.current_basis and scores[j] > best_score:
                best_score = scores[j]
                entering_variable = j
        
        return entering_variable

    def select_leaving_variable(self, entering_variable):
        """
        Select the leaving variable based on the ratio test and optimal basis.

        Args:
        entering_variable : int
            Index of the entering variable.

        Returns:
        int
            Index of the leaving variable.
        """
        B_inv = np.linalg.inv(self.A[:, self.current_basis])
        aj_hat = B_inv @ self.A[:, entering_variable]
        x_b = B_inv @ self.b
        ratios = np.zeros(self.m)
        for i in range(self.m):
            if i in self.current_basis:
                if aj_hat[i] > 0:
                    ratios[i] = x_b[i] / aj_hat[i]
                else:
                    ratios[i] = float('inf')

        min_ratio = float('inf')
        leaving_variable = -1

        for i in range(self.m):
            var = self.current_basis[i]
            if var in self.optimal_basis:
                continue  # Skip variables as it is in optimal basis
            if var in self.current_basis and ratios[i] < min_ratio:
                min_ratio = ratios[i]
                leaving_variable = var

        return leaving_variable

    def simplex_step(self):
        """
        Perform a single step of the Simplex method using the expert's logic.

        Returns:
        tuple
            Index of the entering variable, index of the leaving variable.
        """
        entering_variable = self.select_entering_variable()
        if entering_variable == -1:
            return -1, -1  # Indicate no valid entering variable found
        
        leaving_variable = self.select_leaving_variable(entering_variable)
        if leaving_variable == -1:
            return entering_variable, -1  # Indicate no valid leaving variable found
        
        # Update current basis
        self.current_basis.remove(leaving_variable)
        self.current_basis.append(entering_variable)
        
        return entering_variable, leaving_variable

"""
tolerance = 1e-9
in_basis = [1 if v > tolerance else 0 for v in result.x]
slack_in_basis = [1 if s >= tolerance else 0 for s in result.slack]
in_basis = in_basis[:500]


in_basis = np.array(in_basis)
slack_in_basis = np.array(slack_in_basis)

optimal_basis = np.concatenate((in_basis, slack_in_basis))

basis_indices = np.where(optimal_basis == 1)[0].tolist()
print(len(basis_indices))

optimal_solution = data['optimal_solution']
slack_variables = data['slack_variables']

#Need to get the index
optimal_solution = np.array(optimal_solution)
slack_variables = np.array(slack_variables)

# Concatenate the arrays
optimal_basis = np.concatenate((optimal_solution, slack_variables))
print(optimal_basis)

data2 = np.load(init_basis_folder)

current_basis = data['initial_basis']





# Example usage:
c = np.array([2, 3, -1, -4, 0, 0])
A = np.array([
    [1, 1, 1, 0, 1, 0],
    [2, 0.5, 0, 1, 0, 1]
])
b = np.array([6, 8])
optimal_basis = [2, 3]  # Example optimal basis indices
current_basis = [4, 5]  # Example current basis indices

expert = Expert1(c, A, b, optimal_basis, current_basis)
entering_var, leaving_var = expert.simplex_step()
print(f"Entering variable: {entering_var}, Leaving variable: {leaving_var}")
"""