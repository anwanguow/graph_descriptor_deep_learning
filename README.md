The Deep Learning Implementation of "Graph-based Descriptors for Condensed Matter"
==============



This repository contains the complete set of algorithms and computational data from the paper "Graph-based Descriptors for Condensed Matter". It specifically focuses on the deep learning implementations of the four key experiments presented in the paper.

The preprint edition is available at https://drive.google.com/file/d/1SP3bxhoXYQ117zxxdf5iud5gAEc-1Zj4/view?usp=sharing.

We replaced the traditional machine learning methods mentioned in the paper with a deep learning approach, specifically using GraphSAGE. The graph structure is implemented using the modified Voronoi method as described in the paper, with the parameter set to A=0.55, which was proven to be optimal in traditional machine learning methods. Note that message passing for each sample is only performed within its corresponding graph structure, while all GNNs share a common weight matrix.

For detailed settings, please refer to "GNN_settings.pdf".

The main repository of this article is https://github.com/anwanguow/GP_structural.

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
2. Exp_B: "results/task_2.txt".
3. Exp_C: "results/task_3.txt".
4. Exp_D: "results/task_4.txt".

Contact
-----------------
An Wang: amturing@outlook.com
