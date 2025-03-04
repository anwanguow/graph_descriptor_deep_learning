The Deep Learning Implementation of "Graph-based Descriptors for Condensed Matter"
==============



This repository contains the complete set of algorithms and computational data from the paper "Graph Theory-Based Approach to Identifying Phase Transitions in Condensed Matter." It specifically focuses on the deep learning implementations of the four key experiments presented in the paper, demonstrating how graph-based descriptors can be leveraged to analyze phase transitions in condensed matter systems.

This preprint is available at https://arxiv.org/abs/2408.06156.

We replaced the traditional machine learning methods mentioned in the paper with a deep learning approach, specifically using GraphSAGE. The graph structure is implemented using the modified Voronoi method as described in the paper, with the parameter set to A=0.55, which was proven to be optimal in traditional machine learning methods. Note that message passing for each sample is only performed within its corresponding graph structure, while all GNNs share a common weight matrix. For detailed settings, please refer to "GNN_settings.pdf".

MD simulation
-----------------

All trajectories are stored in the "data/Traj" directory, with 20 trajectories and their corresponding groups organized into subdirectories. Each subdirectory contains LAMMPS scripts for molecular dynamics simulations and the resulting DCD files.


Dataset Generation
-----------------

1. Generation of graph structure: "data/graph_gen.py".

2. Generation of original input features: "data/X_1.py" and "data/X_2.py".

3. Dataset for Exp_A: "data/data_task_1.py".

4. Dataset for Exp_B: "data/data_task_2.py".

5. Dataset for Exp_C: "data/data_task_3.py".

6. Dataset for Exp_D: "data/data_task_4.py".



All python scripts in "Plot" directory reproduces the computed figures in the article, including Fig.2b), Fig.2c), Fig.3a), Fig.3b), Fig.4a), Fig.4b), Fig.5b), Fig.5c), Fig.8a) and Fig.8b). Besides, the generated figures are saved in "Figure" directory, demonstrated as follow:




Training
-----------------
1. Exp_A: "Task_1_train.py".
2. Exp_B: Exp_B use the same optimal model as Exp_A for testing.
3. Exp_C: "Task_3_train.py".
4. Exp_D: "Task_4_train.py".

Testing
-----------------
1. Exp_A: "Task_1_test.py".
2. Exp_B: "Task_2_test.py".
3. Exp_C: "Task_3_test.py".
4. Exp_D: "Task_4_test.py".

Results
-----------------
1. Exp_A: "results/task_1.txt".
1. Exp_B: "results/task_2.txt".
1. Exp_C: "results/task_3.txt".
1. Exp_D: "results/task_4.txt".

