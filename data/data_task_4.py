import numpy as np
import os
import random
import csv

def load_labels(set_id, traj_id, frame_id):
    file_path = f"p_hop_t/set_{set_id}/D_{traj_id}/Y_{frame_id}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        return None

def sample_data(set_id, traj_id, frames, num_samples_each, existing_samples):
    positive_samples = []
    negative_samples = []
    
    for frame_id in frames:
        labels = load_labels(set_id, traj_id, frame_id)
        if labels is None:
            continue
        
        for particle_id, label in enumerate(labels):
            sample_key = (set_id, traj_id, frame_id, particle_id)
            if sample_key in existing_samples:
                continue  
            
            if label == 1:
                positive_samples.append((set_id, traj_id, frame_id, particle_id, label))
            elif label == 0:
                negative_samples.append((set_id, traj_id, frame_id, particle_id, label))
    
    sampled_positive = random.sample(positive_samples, min(num_samples_each, len(positive_samples)))
    sampled_negative = random.sample(negative_samples, min(num_samples_each, len(negative_samples)))
    
    return sampled_positive + sampled_negative

def save_to_csv(filename, samples):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["group", "traj", "frame", "particle", "label"])
        writer.writerows(samples)

groups = [1, 2]
trajectories = list(range(10))
frames = [100, 200, 300, 400, 500, 600, 700, 800, 900]

train_samples = sample_data(set_id=1, traj_id=4, frames=frames, num_samples_each=2500, existing_samples=set())

used_samples = set((s[0], s[1], s[2], s[3]) for s in train_samples)

test_samples = []
for set_id in groups:
    for traj_id in trajectories:
        if set_id == 1 and traj_id == 4:
            continue 
        test_samples.extend(sample_data(set_id, traj_id, frames, num_samples_each=3000, existing_samples=used_samples))
        if len(test_samples) >= 60000:
            break
    if len(test_samples) >= 60000:
        break

test_samples = test_samples[:60000] 

os.makedirs("dataset_index", exist_ok=True)
save_to_csv("dataset_index/Task_4_train.csv", train_samples)
save_to_csv("dataset_index/Task_4_test.csv", test_samples)

print("Done")