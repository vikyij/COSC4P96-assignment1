import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def get_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_dir() -> str:
    return os.path.join(get_repo_root(), "data")


def load_stage1_artifacts(data_dir: str | None = None):
    if data_dir is None:
        data_dir = get_data_dir()

    loaded = np.load(os.path.join(data_dir, "data_splits.npz"))

    with open(os.path.join(data_dir, "normalization_stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    mean = stats["mean"]
    std = stats["std"]
    return loaded, mean, std


def build_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])


def load_full_cifar10(data_dir: str | None = None):
    """
    Returns full_data as numpy array of shape (60000, 32, 32, 3)
    """
    if data_dir is None:
        data_dir = get_data_dir()

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    testset  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    return np.concatenate((trainset.data, testset.data), axis=0)


class CIFAR10Subset(Dataset):
    def __init__(self, full_data, indices, targets, transform=None):
        self.full_data = full_data
        self.indices = indices
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = int(self.indices[idx])
        image = self.full_data[original_idx]
        label = int(self.targets[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


def make_loaders(batch_size=64, shuffle_labeled=True):
    loaded, mean, std = load_stage1_artifacts()
    full_data = load_full_cifar10()
    transform = build_transform(mean, std)
    unlabeled_indices = loaded['unlabeled_indices']

    labeled_dataset = CIFAR10Subset(full_data, loaded["labeled_indices"], loaded["y_labeled"], transform)
    val_dataset     = CIFAR10Subset(full_data, loaded["val_indices"], loaded["y_val"], transform)
    test_dataset    = CIFAR10Subset(full_data, loaded["test_indices"], loaded["y_test"], transform)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=shuffle_labeled)
    val_loader     = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader    = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return labeled_dataset, val_dataset, test_dataset, labeled_loader, val_loader, test_loader, full_data, unlabeled_indices


def initialize_weights(model, init_type="xavier", activation="relu"):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            fan_in = m.weight.size(1)

            if init_type == "uniform":
                limit = 1.0 / np.sqrt(fan_in)
                nn.init.uniform_(m.weight, -limit, limit)

            elif init_type == "normal":
                nn.init.normal_(m.weight, mean=0, std=0.01)

            elif init_type == "xavier":
                if activation.lower() == "relu":
                    # He initialization for ReLU
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    # Xavier/Glorot initialization for Sigmoid/Tanh
                    nn.init.xavier_uniform_(m.weight)

            else:
                raise ValueError(f"Unknown init_type: {init_type}")
            
            # Initialize bias to zero
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class EarlyStopping:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def check_stop(self, current_val_loss):
        if len(self.history) < self.window_size:
            self.history.append(current_val_loss)
            return False

        mean_loss = float(np.mean(self.history))
        std_loss = float(np.std(self.history))
        stop = current_val_loss > (mean_loss + std_loss)

        self.history.append(current_val_loss)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        return stop
