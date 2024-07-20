# Exploring-Linear-Programs-with-Graphs-and-AI
This is a code repository for the attempted survey paper:  Exploring Linear Programs with Graphs and AI. It outlines the code experimentation on Smart Iniital Basis and Learn To Pivot model for an AI Simplex Method

The files are arranged as followed in order:

FOR SIB:\
generator.py: Generate LP instances.\
SIB/bipartite_transform.py: Transform LP instances into bipartite graphs.\
SIB/arch.py: Architecture for SIB.\
SIB/runner2.py (and runner3.py): Trains SIB for primal and slack variables.\
SIB/test.py: Retrieve the probability of the test dataset from SIB models. \
SIB/unpack.py: Turn the probability for each basis into a proper basis of 1 and 0s.\

FOR LTP:\\
LTP/expert.py: Contains the pivoting expert that pivots an initial basis to the optimal value.\
LTP/collect_exp_choice.py: Generate a file that contains expert pivot choices.\
LTP/bipartite_pivot_transform.py: Transform LP instances with expert choice as labels.\
LTP/pivot_arch.py: Architecture GCN used for LTP model.\
LTP/pivot_runner.py: Trains the LTP model.\
system_test.py: Test the entire system ot SBI + LTP and save the number of correct 1's (in optimal basis).\
