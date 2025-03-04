import os
import numpy as np
import pandas as pd

base_path = "p_hop_t/set_1"
output_dir = "dataset_index"
os.makedirs(output_dir, exist_ok=True)

trajectory_ids = range(10)
frames = [100, 200, 300, 400, 500, 600, 700, 800, 900]

def load_labels():
    samples = []
    for traj_id in trajectory_ids:
        for frame in frames:
            file_path = os.path.join(base_path, f"D_{traj_id}", f"Y_{frame}.npy")
            if os.path.exists(file_path):
                labels = np.load(file_path)
                for particle, label in enumerate(labels):
                    samples.append([1, traj_id, frame, particle, label])
    return pd.DataFrame(samples, columns=["group", "traj", "frame", "particle", "label"])

def sample_balanced_data(df, num_samples):
    pos_samples = df[df["label"] == 1].sample(n=num_samples//2, random_state=42)
    neg_samples = df[df["label"] == 0].sample(n=num_samples//2, random_state=42)
    return pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

data = load_labels()
train_data = sample_balanced_data(data, 15000)
remaining_data = data.drop(train_data.index)
test_data = sample_balanced_data(remaining_data, 15000)

train_data.to_csv(os.path.join(output_dir, "Task_1_train.csv"), index=False)
test_data.to_csv(os.path.join(output_dir, "Task_1_test.csv"), index=False)

print("Done.")
