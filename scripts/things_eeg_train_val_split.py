import numpy as np
import random
import os

random.seed(42)  # For reproducibility
np.random.seed(42)  # For reproducibility

# Load the data
data = np.load('/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2/images/image_metadata.npy', allow_pickle=True).item()
save_path = "/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2/images/"
train_concepts = data['train_img_concepts']  # This has 16540 elements (1654 concepts × 10)
train_files = data['train_img_files']

# Constants
samples_per_concept = 10
num_total_concepts = len(train_concepts) // samples_per_concept
concept_indices = list(range(num_total_concepts))

# Randomly select 80% of concepts
num_selected_concepts = int(0.8 * num_total_concepts)
selected_concept_ids = sorted(random.sample(concept_indices, num_selected_concepts))  # sorted for convenience
excluded_concept_ids = sorted(set(concept_indices) - set(selected_concept_ids))

# Get all indices corresponding to selected concepts
selected_indices = []
for concept_id in selected_concept_ids:
    base_idx = concept_id * samples_per_concept
    selected_indices.extend(list(range(base_idx, base_idx + samples_per_concept)))

# Get sample indices for excluded (validation) concepts
excluded_indices = []
for cid in excluded_concept_ids:
    base_idx = cid * samples_per_concept
    excluded_indices.extend(range(base_idx, base_idx + samples_per_concept))

# Create new metadata dict with selected subset
selected_metadata = {
    'img_concepts': [train_concepts[i] for i in selected_indices],
    'img_files':    [train_files[i] for i in selected_indices],
    'indices':      selected_indices  # store the original indices
}

# Save to a new file
np.save(os.path.join(save_path, 'image_metadata_train.npy'), selected_metadata)

# --- Create and save excluded (val) metadata ---
excluded_metadata = {
    'img_concepts': [train_concepts[i] for i in excluded_indices],
    'img_files':    [train_files[i] for i in excluded_indices],
    'indices':      excluded_indices
}
np.save(os.path.join(save_path, 'image_metadata_val.npy'), excluded_metadata)

# Extract test data
test_concepts = data['test_img_concepts']
test_files = data['test_img_files']
test_indices = list(range(len(test_concepts)))  # Just 0...N-1

# Create and save test metadata
test_metadata = {
    'img_concepts': test_concepts,
    'img_files': test_files,
    'indices': test_indices
}
np.save(os.path.join(save_path, 'image_metadata_test.npy'), test_metadata)

print(f"Saved {len(selected_concept_ids)} concepts ({len(selected_indices)} samples) to 'train_subset_metadata.npy'")