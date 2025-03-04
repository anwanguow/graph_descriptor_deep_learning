#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

base_path = "p_hop_t/set_2"
output_csv = "dataset_index/Task_2_test.csv"
sample_size = 30000

samples = []

for traj_id in range(10):
    traj_path = os.path.join(base_path, f"D_{traj_id}")
    
    for frame in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
        label_file = os.path.join(traj_path, f"Y_{frame}.npy")
        
        if os.path.exists(label_file):
            labels = np.load(label_file)
            
            for particle_idx, label in enumerate(labels):
                samples.append([2, traj_id, frame, particle_idx, label])

df = pd.DataFrame(samples, columns=["group", "traj", "frame", "particle", "label"])

pos_samples = df[df["label"] == 1]
neg_samples = df[df["label"] == 0]

min_samples = min(len(pos_samples), len(neg_samples), sample_size // 2)

pos_samples = pos_samples.sample(n=min_samples, random_state=42)
neg_samples = neg_samples.sample(n=min_samples, random_state=42)

balanced_df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

balanced_df.to_csv(output_csv, index=False)

print("Done")
