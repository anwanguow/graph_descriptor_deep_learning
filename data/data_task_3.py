import os
import numpy as np
import pandas as pd

LABEL_PATH = "p_hop_t/set_{group}/D_{traj}/Y_{frame}.npy"

groups = [1, 2]
traj_ids = list(range(10))
frames_train = [100]
frames_test = [200, 300, 400, 500, 600, 700, 800, 900]

train_sample_size = 10000
test_sample_size = 30000

def load_labels(group, traj, frame):
    path = LABEL_PATH.format(group=group, traj=traj, frame=frame)
    return np.load(path)

def sample_balanced(all_labels, sample_size):
    pos_indices = np.where(all_labels == 1)[0]
    neg_indices = np.where(all_labels == 0)[0]
    
    pos_sample_size = sample_size // 2
    neg_sample_size = sample_size // 2
    
    if len(pos_indices) < pos_sample_size or len(neg_indices) < neg_sample_size:
        raise ValueError("Data is not enough")
    
    pos_samples = np.random.choice(pos_indices, pos_sample_size, replace=False)
    neg_samples = np.random.choice(neg_indices, neg_sample_size, replace=False)
    
    return np.concatenate((pos_samples, neg_samples))

def collect_all_labels(frames):
    all_labels = []
    all_metadata = []
    
    for group in groups:
        for traj in traj_ids:
            for frame in frames:
                labels = load_labels(group, traj, frame)
                for i, label in enumerate(labels):
                    all_labels.append(label)
                    all_metadata.append([group, traj, frame, i])
    
    return np.array(all_labels), all_metadata

train_labels, train_metadata = collect_all_labels(frames_train)
train_sampled_indices = sample_balanced(train_labels, train_sample_size)
train_data = [train_metadata[i] + [train_labels[i]] for i in train_sampled_indices]

test_labels, test_metadata = collect_all_labels(frames_test)
test_sampled_indices = sample_balanced(test_labels, test_sample_size)
test_data = [test_metadata[i] + [test_labels[i]] for i in test_sampled_indices]

df_train = pd.DataFrame(train_data, columns=["group", "traj", "frame", "particle", "label"])
df_test = pd.DataFrame(test_data, columns=["group", "traj", "frame", "particle", "label"])

os.makedirs("dataset_index", exist_ok=True)
df_train.to_csv("dataset_index/Task_3_train.csv", index=False)
df_test.to_csv("dataset_index/Task_3_test.csv", index=False)

print("Done.")
