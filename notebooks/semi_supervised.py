import os,sys
import time
import torch.optim as optim
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_stage1_artifacts, initialize_weights, EarlyStopping, make_loaders
loaded, mean, std = load_stage1_artifacts()
labeled_dataset, _, _,labeled_loader, val_loader, test_loader, full_data, unlabeled_indices = make_loaders(batch_size=64)


# Define a separate transforms.Compose pipeline for data augmentation
# Using loaded mean and std statistics from the environment
consistency_transform = transforms.Compose([
    transforms.ToPILImage(), # Convert numpy array to PIL Image for transforms
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

print("Consistency data augmentation pipeline (consistency_transform) defined successfully.")



# 1. Define custom Dataset class for unlabeled data
class CIFAR10UnlabeledSubset(Dataset):
    def __init__(self, full_data, indices, transform=None):
        self.full_data = full_data
        self.indices = indices
        self.transform = transform # This transform will be applied to x_prime

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Retrieve original image using the stored index
        original_idx = self.indices[idx]
        image_data = self.full_data[original_idx] # Returns as numpy array (H, W, C)

        # For unlabeled data, we might need both the original (x) and augmented (x_prime) version
        # The original image data will be handled separately in the training loop for x
        # The transform here will create x_prime
        if self.transform:
            x_prime = self.transform(image_data)
        else:
            # If no transform specified, return the original data as a tensor
            x_prime = torch.from_numpy(image_data).permute(2, 0, 1).float() / 255.0 # Basic normalization if no transform

        return image_data, x_prime # Return original numpy array and transformed tensor

# 2. Instantiate the custom unlabeled dataset
unlabeled_dataset = CIFAR10UnlabeledSubset(full_data, unlabeled_indices, transform=consistency_transform)

# 3. Create an unlabeled_loader. Match batch_size with labeled_loader.
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=labeled_loader.batch_size, shuffle=True)

print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
print("Unlabeled DataLoader created: unlabeled_loader")


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout_rate=0.0):
        super(FeedForwardNN, self).__init__()

        layers = []

        # Determine activation function
        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation.lower() == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build hidden layers
        current_dim = input_size
        for hidden_dim in hidden_sizes:
            # nn.Linear defaults to bias=True
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(act_fn)

            # Add dropout layer if rate > 0
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            current_dim = hidden_dim

        # Output layer (no activation, returns logits for CrossEntropyLoss)
        layers.append(nn.Linear(current_dim, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input: [batch, C, H, W] -> [batch, input_size]
        if x.dim() > 2:
            # Changed .view to .reshape to handle non-contiguous tensors
            x = x.reshape(x.size(0), -1)

        return self.network(x)

def run_stage4_experiment(seed, input_size=3072, hidden_sizes=[256, 128], output_size=10,
                          activation='relu', lr=0.01, momentum=0.9, max_epochs=50,
                          init_type='xavier', dropout_rate=0.0, weight_decay=0.0,
                          lambda_consistency=0.0, consistency_warmup_epochs=0):
    # a. Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Stage 4 experiment with Seed {seed}, Labeled_consistency {lambda_consistency}, Warmup {consistency_warmup_epochs} on {device}...")

    # b. Instantiate Model
    model = FeedForwardNN(input_size, hidden_sizes, output_size, activation, dropout_rate).to(device)

    # c. Initialize Weights
    initialize_weights(model, init_type=init_type, activation=activation)

    # d. Define Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # e. Instantiate Early Stopping
    early_stopping = EarlyStopping(window_size=5)

    # f. Initialize metrics storage
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'consistency_loss': [] 
    }

    # g. Training Loop
    start_time = time.time()
    stopped_epoch = max_epochs

    # Create a normalization-only transform to apply to inputs_ul (original unlabeled images)
    normalize_only = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        running_consistency_loss = 0.0
        correct_train = 0
        total_train = 0

        # Create iterators for both loaders
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        # Determine the number of batches to process in this epoch - iterate as long as there is labeled data, and try to match with unlabeled batches
        num_batches = len(labeled_loader)

        for batch_idx in range(num_batches):
            # Get labeled batch
            try:
                inputs_l, labels_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_l, labels_l = next(labeled_iter)
            inputs_l, labels_l = inputs_l.to(device), labels_l.to(device)

            # Get unlabeled batch
            try:
                inputs_ul_raw, inputs_ul_prime = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                inputs_ul_raw, inputs_ul_prime = next(unlabeled_iter)
            
            # Convert to (batch_size, C, H, W) and scale to [0, 1]
            inputs_ul = inputs_ul_raw.permute(0, 3, 1, 2).float() / 255.0

            # Apply normalization
            inputs_ul = normalize_only(inputs_ul)

            inputs_ul = inputs_ul.to(device)
            inputs_ul_prime = inputs_ul_prime.to(device)

            optimizer.zero_grad()

            # Forward pass for labeled data
            outputs_l = model(inputs_l)
            loss_l = criterion(outputs_l, labels_l)

            # Forward pass for unlabeled data (for consistency loss)
            # Treat p_theta(y|x_j) as a fixed target (stop gradients for x_j in the first pass)
            with torch.no_grad(): 
                targets_ul = F.softmax(model(inputs_ul), dim=1)

            outputs_ul_prime = model(inputs_ul_prime)
            log_softmax_outputs_ul_prime = F.log_softmax(outputs_ul_prime, dim=1)
            
         
            loss_consistency = (targets_ul * torch.log(targets_ul + 1e-10) - targets_ul * log_softmax_outputs_ul_prime).sum(dim=1).mean()

            # Warm-up schedule for lambda_consistency
            current_lambda = lambda_consistency * min(epoch / consistency_warmup_epochs, 1.0) if consistency_warmup_epochs > 0 else lambda_consistency

            # Total loss
            total_loss = loss_l + current_lambda * loss_consistency

            # Backpropagation and optimization
            total_loss.backward() 
            optimizer.step()

            running_loss += loss_l.item() * inputs_l.size(0)
            running_consistency_loss += loss_consistency.item() * inputs_ul.size(0)

            _, predicted = torch.max(outputs_l.data, 1)
            total_train += labels_l.size(0)
            correct_train += (predicted == labels_l).sum().item()

        epoch_train_loss = running_loss / total_train if total_train > 0 else 0
        epoch_train_acc = correct_train / total_train if total_train > 0 else 0
        epoch_consistency_loss = running_consistency_loss / len(unlabeled_dataset) if len(unlabeled_dataset) > 0 else 0

        # --- Validation Phase (same as before) ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / total_val if total_val > 0 else 0
        epoch_val_acc = correct_val / total_val if total_val > 0 else 0

        # Store metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['consistency_loss'].append(epoch_consistency_loss)

        # Check Early Stopping
        if early_stopping.check_stop(epoch_val_loss): # Early stopping still based on validation loss
            print(f"Early stopping triggered at epoch {epoch+1}")
            stopped_epoch = epoch + 1
            break

    total_time = time.time() - start_time

    # i. Test Phase 
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = correct_test / total_test if total_test > 0 else 0

    return history, total_time, stopped_epoch, test_accuracy

print("run_stage4_experiment function defined successfully.")

# Instantiate a test model to verify architecture
# CIFAR-10 images: 32x32x3 = 3072 features
input_size = 3072
hidden_sizes = [256, 128]
output_size = 10
activation = 'relu'

model = FeedForwardNN(input_size, hidden_sizes, output_size, activation)
print("Model Architecture:")
print(model)

seeds = [1, 123, 12345]
lambda_consistency = 1.0
consistency_warmup_epochs = 10

consistency_results = {}

print("Starting Consistency Regularization Experiments...\n")

for seed in seeds:
    print(f"--- Consistency Regularization Experiment with Seed {seed} ---")
    hist, time_taken, epochs, test_acc = run_stage4_experiment(
        seed,
        lambda_consistency=lambda_consistency,
        consistency_warmup_epochs=consistency_warmup_epochs
    )
    consistency_results[seed] = {
        'history': hist,
        'time': time_taken,
        'epochs': epochs,
        'test_acc': test_acc
    }
    print(f"Seed {seed} Completed: Test Acc: {test_acc:.4f}, Epochs: {epochs}, Time: {time_taken:.2f}s\n")

print("Consistency Regularization experiments completed.")

# 1. Setup the figure for plotting
num_seeds = len(consistency_results)
fig, axes = plt.subplots(num_seeds, 2, figsize=(15, 5 * num_seeds))

# Ensure axes is 2D array even if num_seeds is 1
if num_seeds == 1:
    axes = np.array([axes])

consistency_summary_data = []
consistency_test_accuracies = []

# 2. Iterate through results and plot
for i, (seed, data) in enumerate(sorted(consistency_results.items())):
    history = data['history']
    epochs_range = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    ax_loss = axes[i, 0]
    ax_loss.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    ax_loss.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
    ax_loss.set_title(f'Seed {seed} (Consistency Reg.): Loss vs Epochs')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)

    # Plot Accuracy
    ax_acc = axes[i, 1]
    ax_acc.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
    ax_acc.plot(epochs_range, history['val_acc'], label='Val Acc', marker='o')
    ax_acc.set_title(f'Seed {seed} (Consistency Reg.): Accuracy vs Epochs')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)

    # Collect data for summary
    consistency_summary_data.append({
        'Seed': seed,
        'Test Accuracy': data['test_acc'],
        'Epochs to Converge': data['epochs'],
        'Time (s)': data['time']
    })
    consistency_test_accuracies.append(data['test_acc'])

plt.tight_layout()
plt.show()

# 3. Create Summary DataFrame
consistency_summary_df = pd.DataFrame(consistency_summary_data)

# 4. Calculate Statistics
consistency_mean_acc = np.mean(consistency_test_accuracies)
consistency_std_acc = np.std(consistency_test_accuracies)

# 5. Print Report
print("\n--- Consistency Regularization Performance Summary ---")
print(consistency_summary_df.to_string(index=False))
print("\n--- Aggregate Statistics ---")
print(f"Mean Test Accuracy: {consistency_mean_acc:.4f}")
print(f"Std Dev Test Accuracy: {consistency_std_acc:.4f}")

rows = []

for seed in seeds:
    hist, time_taken, epochs, test_acc = run_stage4_experiment(
        seed,
        lambda_consistency=lambda_consistency,
        consistency_warmup_epochs=consistency_warmup_epochs
    )

    rows.append({
        "seed": seed,
        "test_accuracy": test_acc,
        "epochs_to_converge": epochs,
        "training_time_seconds": time_taken,
        "lr": 0.01,
        "momentum": 0.9,
        "init_type": "xavier",
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "lambda_consistency": lambda_consistency,
        "consistency_warmup_epochs": consistency_warmup_epochs,
        "consistency_mean_acc": consistency_mean_acc,
        "consistency_std_acc": consistency_std_acc
    })

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("results/semi-supervised.csv", index=False)
