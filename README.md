# Exploring Linear Programs with Graphs and AI

# IPMs-MPNN
For those who are interested in our experimentation for IPM-MPNN on the combined dataset that contains: Setcover, Indset, Facilities, and Cauction, please refer to (Qian et al., 2023). Our code was a direct replication of theirs except we altered the dataset to contain all four problem type.

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/boomer3boom/Exploring-Linear-Programs-with-Graphs-and-AI/blob/main/Images/IPM/objgap.png" alt="Validation Plot" width="400" style="margin-right: 10px;"/>
  <img src="https://github.com/boomer3boom/Exploring-Linear-Programs-with-Graphs-and-AI/blob/main/Images/IPM/consgap.png" alt="Train Plot" width="400"/>
</div>

# Ensemble Model of SIB and LTP
This is a code repository for the attempted survey paper: **Exploring Linear Programs with Graphs and AI**. It outlines the code experimentation on Smart Initial Basis (SIB) and Learn To Pivot (LTP) model for an AI Simplex Method. **Please note this paper is not published and was a student attempt during research internship.**

This framework is arranged as:
1. Generate LP Instance
2. Train SIB for initial basis
3. Use initial basis and collect pivoting expert's pivot
4. Train LTP through imitation learning
5. Test the framework

## Bipartite Graph Features

### Training SIB
For Variable nodes:
- Feature 1: Objective coefficient
- Feature 2: nnz(A) / num_constraints

For Constraint nodes:
- Feature 1: RHS constraint coefficient
- Feature 2: nnz(A) / num_variables

### Training LTP
For Variable nodes:
- Feature 1: Objective coefficient
- Feature 2: nnz(A) / num_constraints
- Feature 3: If variable in current basis (1) or not (0)

For Constraint nodes:
- Feature 1: RHS constraint coefficient
- Feature 2: nnz(A) / num_variables
- Feature 3: If the corresponding slack variable in current basis (1) or not (0)

## Project Structure

### FOR SIB:
- `generator.py`: Generate LP instances.
- `SIB/bipartite_transform.py`: Transform LP instances into bipartite graphs.
- `SIB/arch.py`: Architecture for SIB.
- `SIB/runner2.py` (and `runner3.py`): Trains SIB for primal and slack variables.
- `SIB/test.py`: Retrieve the probability of the test dataset from SIB models.
- `SIB/unpack.py`: Turn the probability for each basis into a proper basis of 1 and 0s.

### FOR LTP:
- `LTP/expert.py`: Contains the pivoting expert that pivots an initial basis to the optimal value.
- `LTP/collect_exp_choice.py`: Generate a file that contains expert pivot choices.
- `LTP/bipartite_pivot_transform.py`: Transform LP instances with expert choice as labels.
- `LTP/pivot_arch.py`: Architecture GCN used for LTP model.
- `LTP/pivot_runner.py`: Trains the LTP model.
- `system_test.py`: Test the entire system of SIB + LTP and save the number of correct 1's (in optimal basis).

## Existing Model:
- `primal.pth`: Our existing primal SIB model.

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/boomer3boom/Exploring-Linear-Programs-with-Graphs-and-AI/blob/main/Images/SIB%20Primal/validation.png" alt="Validation Plot" width="400" style="margin-right: 10px;"/>
  <img src="https://github.com/boomer3boom/Exploring-Linear-Programs-with-Graphs-and-AI/blob/main/Images/SIB%20Primal/train.png" alt="Train Plot" width="400"/>
</div>

- `slack.pth`: Our existing slack SIB model.

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/boomer3boom/Exploring-Linear-Programs-with-Graphs-and-AI/blob/main/Images/SIB%20Slack/validation.png" alt="Validation Plot" width="400" style="margin-right: 10px;"/>
  <img src="https://github.com/boomer3boom/Exploring-Linear-Programs-with-Graphs-and-AI/blob/main/Images/SIB%20Slack/train.png" alt="Train Plot" width="400"/>
</div>

- `pivot_learner.pth`: Our existing LTP model.

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/boomer3boom/Exploring-Linear-Programs-with-Graphs-and-AI/blob/main/Images/pivot%20model%20preformance/validation.png" alt="Validation Plot" width="400" style="margin-right: 10px;"/>
  <img src="https://github.com/boomer3boom/Exploring-Linear-Programs-with-Graphs-and-AI/blob/main/Images/pivot%20model%20preformance/train.png" alt="Train Plot" width="400"/>
</div>
