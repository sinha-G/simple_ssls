# datasets/dataset_utils.py
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def split_dataset(dataset, num_labeled):
    # Extract labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # Perform stratified sampling
    labeled_idx, unlabeled_idx = train_test_split(
        np.arange(len(dataset)),
        train_size=num_labeled,
        stratify=labels,
        random_state=42  # For reproducibility
    )

    # Subset the dataset
    labeled_data = torch.utils.data.Subset(dataset, labeled_idx)
    unlabeled_data = torch.utils.data.Subset(dataset, unlabeled_idx)

    return labeled_data, unlabeled_data