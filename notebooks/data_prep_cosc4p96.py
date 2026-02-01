# -*- coding: utf-8 -*-
"""
# From Supervised to Semi-Supervised Learning A Complete Machine Learning Pipeline

Stage 1: Data Preparation and Preprocessing.

## Project Setup

Initialize the environment, Import necessary libraries and set random seeds.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import torchvision
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

set_seed(42)

"""## Data Loading and Partitioning

Load the complete CIFAR-10 dataset and partition it into Test, Validation, Labeled, and Unlabeled sets according to the specified percentages.
"""

# Load CIFAR-10 dataset
train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Concatenate data and targets to form a single collection
X = np.concatenate([train_ds.data, test_ds.data], axis=0)
y = np.concatenate([train_ds.targets, test_ds.targets], axis=0)

print(f"Total dataset size: {len(y)}")

# 1. Split Test Set (10% of 60,000 = 6,000)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=6000, stratify=y, random_state=42
)

# 2. Split Validation Set (10% of 60,000 = 6,000)
X_train_pool, X_val, y_train_pool, y_val = train_test_split(
    X_temp, y_temp, test_size=6000, stratify=y_temp, random_state=42
)

# 3. Split Training Pool (48,000) into Labeled (20%) and Unlabeled (80%)
# test_size=0.2 means 20% goes to the second output (Labeled)
X_unlabeled, X_labeled, y_unlabeled, y_labeled = train_test_split(
    X_train_pool, y_train_pool, test_size=0.2, stratify=y_train_pool, random_state=42
)

# Verify the counts
print(f"Test Set: {len(y_test)} samples")
print(f"Validation Set: {len(y_val)} samples")
print(f"Labeled Set: {len(y_labeled)} samples")
print(f"Unlabeled Set: {len(y_unlabeled)} samples")

"""## Preprocessing Implementation and Analysis

Implement Min-Max scaling and Z-score normalization on the training pool data, calculate statistics from the training pool, and analyze feature ranges.

"""

# 1. Original Data Statistics
# Convert to float for accurate statistical calculation
X_train_float = X_train_pool.astype(np.float32)

print("Original Data Statistics (0-255 range):")
print(f"  Min: {X_train_float.min():.2f}")
print(f"  Max: {X_train_float.max():.2f}")
print(f"  Mean: {X_train_float.mean():.2f}")
print(f"  Std: {X_train_float.std():.2f}")
print("-" * 30)

# 2. Implement Min-Max Scaling to [0, 1]
# Simple division by 255 for image data
X_train_minmax = X_train_float / 255.0

print("Min-Max Scaled Statistics ([0, 1] range):")
print(f"  Min: {X_train_minmax.min():.4f}")
print(f"  Max: {X_train_minmax.max():.4f}")
print(f"  Mean: {X_train_minmax.mean():.4f}")
print(f"  Std: {X_train_minmax.std():.4f}")
print("-" * 30)

# 3. Compute Per-Channel Statistics for Z-Score Normalization
# CIFAR-10 is (N, H, W, C). Compute stats over axes (0, 1, 2)
mean = X_train_minmax.mean(axis=(0, 1, 2))
std = X_train_minmax.std(axis=(0, 1, 2))

print("Computed Per-Channel Statistics (from Training Pool):")
print(f"  Mean (R, G, B): {mean}")
print(f"  Std  (R, G, B): {std}")
print("-" * 30)

# 4. Implement Z-Score Normalization
# (X - mean) / std
X_train_zscore = (X_train_minmax - mean) / std

print("Z-Score Normalized Statistics (Zero Mean, Unit Variance):")
print(f"  Min: {X_train_zscore.min():.4f}")
print(f"  Max: {X_train_zscore.max():.4f}")
print(f"  Mean: {X_train_zscore.mean():.4f} (Expected ~0)")
print(f"  Std: {X_train_zscore.std():.4f} (Expected ~1)")

"""## Data Augmentation Visualization

Implement Random Horizontal Flip and Random Crop augmentations and visualize the results on a few sample images to verify their effects.
"""

from torchvision import transforms
from PIL import Image

# Define the augmentation pipeline
# ToPILImage is needed because X_train_pool contains numpy arrays
augment_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])

# Select 5 random indices
indices = np.random.choice(len(X_train_pool), 5, replace=False)

plt.figure(figsize=(8, 12))

for i, idx in enumerate(indices):
    # Get original image (numpy array)
    original_img_np = X_train_pool[idx]

    # Apply augmentation
    # The pipeline converts np array -> PIL -> Apply Transforms
    augmented_img_pil = augment_pipeline(original_img_np)

    # Plot Original
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(original_img_np)
    plt.title(f"Original (Idx {idx})")
    plt.axis('off')

    # Plot Augmented
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(augmented_img_pil)
    plt.title("Augmented")
    plt.axis('off')

plt.tight_layout()
plt.show()

"""## Class Distribution Analysis


Plot the class distributions for the Labeled, Unlabeled, Validation, and Test sets to verify data balance.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_distribution(y_data, title, ax):
    """Helper function to plot class distribution on a given axis."""
    unique, counts = np.unique(y_data, return_counts=True)
    ax.bar(unique, counts, align='center', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(10))
    ax.set_title(f"{title} (Total: {len(y_data)})", fontsize=10)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Count")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# Create a 2x2 grid for plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot distributions
plot_distribution(y_labeled, "Labeled Set", axes[0, 0])
plot_distribution(y_unlabeled, "Unlabeled Set", axes[0, 1])
plot_distribution(y_val, "Validation Set", axes[1, 0])
plot_distribution(y_test, "Test Set", axes[1, 1])

plt.tight_layout()
plt.show()
