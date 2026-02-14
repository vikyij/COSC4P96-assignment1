import numpy as np
import os,sys
import torch
import torch.nn as nn
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import make_loaders, initialize_weights, EarlyStopping

labeled_dataset, val_dataset, test_dataset,labeled_loader, val_loader, test_loader, _, _ = make_loaders(batch_size=64)

print(f"Labeled dataset size: {len(labeled_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print("DataLoaders created: labeled_loader, val_loader, test_loader")

# Modify the `FeedForwardNN` class to incorporate dropout layers based on a configurable dropout rate.

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
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(act_fn)

            # Add dropout layer if rate > 0
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

# Instantiate a test model to verify architecture with dropout
input_size = 3072
hidden_sizes = [256, 128]
output_size = 10
activation = 'relu'
dropout_rate = 0.5

model_dropout = FeedForwardNN(input_size, hidden_sizes, output_size, activation, dropout_rate)
print("Model Architecture with Dropout:")
print(model_dropout)


print("Training utilities (initialize_weights, EarlyStopping) implemented.")

# Run Dropout Experiments

def run_stage3_experiment(seed, input_size=3072, hidden_sizes=[256, 128], output_size=10,
                          activation='relu', lr=0.01, momentum=0.9, max_epochs=50,
                          init_type='xavier', dropout_rate=0.0, weight_decay=0.0):
    # a. Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Stage 3 experiment with Seed {seed}, Dropout {dropout_rate}, Weight Decay {weight_decay}...")

    # b. Instantiate Model (now with dropout_rate)
    model = FeedForwardNN(input_size, hidden_sizes, output_size, activation, dropout_rate).to(device)

    # c. Initialize Weights
    initialize_weights(model, init_type=init_type, activation=activation)

    # d. Define Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer now accepts weight_decay
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # e. Instantiate Early Stopping
    early_stopping = EarlyStopping(window_size=5)

    # f. Initialize metrics storage
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # g. Training Loop
    start_time = time.time()
    stopped_epoch = max_epochs

    for epoch in range(max_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in labeled_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- Validation Phase ---
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

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # Store metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Check Early Stopping
        if early_stopping.check_stop(epoch_val_loss):
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

    test_accuracy = correct_test / total_test

    return history, total_time, stopped_epoch, test_accuracy

# Run Dropout Experiments
seeds = [1, 123, 12345]
dropout_rate = 0.5
# weight_decay stays 0.0 for this specific subtask
dropout_results = {}

print("Starting Dropout Experiments...\n")

for seed in seeds:
    print(f"--- Dropout Experiment with Seed {seed} ---")
    hist, time_taken, epochs, test_acc = run_stage3_experiment(
        seed, dropout_rate=dropout_rate, weight_decay=0.0
    )
    dropout_results[seed] = {
        'history': hist,
        'time': time_taken,
        'epochs': epochs,
        'test_acc': test_acc
    }
    print(f"Seed {seed} Completed: Test Acc: {test_acc:.4f}, Epochs: {epochs}, Time: {time_taken:.2f}s\n")

print("Dropout experiments completed.")

# visualize these results and report the performance metrics

# 1. Setup the figure for plotting
num_seeds = len(dropout_results)
fig, axes = plt.subplots(num_seeds, 2, figsize=(15, 5 * num_seeds))

# Ensure axes is 2D array even if num_seeds is 1
if num_seeds == 1:
    axes = np.array([axes])

dropout_summary_data = []
dropout_test_accuracies = []

# 2. Iterate through results and plot
for i, (seed, data) in enumerate(sorted(dropout_results.items())):
    history = data['history']
    epochs_range = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    ax_loss = axes[i, 0]
    ax_loss.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    ax_loss.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
    ax_loss.set_title(f'Seed {seed} (Dropout 0.5): Loss vs Epochs')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)

    # Plot Accuracy
    ax_acc = axes[i, 1]
    ax_acc.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
    ax_acc.plot(epochs_range, history['val_acc'], label='Val Acc', marker='o')
    ax_acc.set_title(f'Seed {seed} (Dropout 0.5): Accuracy vs Epochs')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)

    # Collect data for summary
    dropout_summary_data.append({
        'Seed': seed,
        'Test Accuracy': data['test_acc'],
        'Epochs to Converge': data['epochs'],
        'Time (s)': data['time']
    })
    dropout_test_accuracies.append(data['test_acc'])

plt.tight_layout()
plt.show()

# 3. Create Summary DataFrame
dropout_summary_df = pd.DataFrame(dropout_summary_data)

# 4. Calculate Statistics
dropout_mean_acc = np.mean(dropout_test_accuracies)
dropout_std_acc = np.std(dropout_test_accuracies)

# 5. Print Report
print("\n--- Dropout (0.5) Performance Summary ---")
print(dropout_summary_df.to_string(index=False))
print("\n--- Aggregate Statistics ---")
print(f"Mean Test Accuracy: {dropout_mean_acc:.4f}")
print(f"Std Dev Test Accuracy: {dropout_std_acc:.4f}")



# Run Weight Decay Experiments Train the model using Weight Decay (L2 regularization) across three random seeds and visualize the results.

seeds = [1, 123, 12345]
weight_decay = 1e-3
# dropout_rate stays 0.0 for this specific subtask
weight_decay_results = {}

print("Starting Weight Decay Experiments...\n")

for seed in seeds:
    print(f"--- Weight Decay Experiment with Seed {seed} ---")
    hist, time_taken, epochs, test_acc = run_stage3_experiment(
        seed, dropout_rate=0.0, weight_decay=weight_decay
    )
    weight_decay_results[seed] = {
        'history': hist,
        'time': time_taken,
        'epochs': epochs,
        'test_acc': test_acc
    }
    print(f"Seed {seed} Completed: Test Acc: {test_acc:.4f}, Epochs: {epochs}, Time: {time_taken:.2f}s\n")

print("Weight Decay experiments completed.")

# Visualize the training and validation curves (loss and accuracy) and report the final performance metrics (test accuracy, epochs, time).

# 1. Setup the figure for plotting
num_seeds = len(weight_decay_results)
fig, axes = plt.subplots(num_seeds, 2, figsize=(15, 5 * num_seeds))

# Ensure axes is 2D array even if num_seeds is 1
if num_seeds == 1:
    axes = np.array([axes])

wd_summary_data = []
wd_test_accuracies = []

# 2. Iterate through results and plot
for i, (seed, data) in enumerate(sorted(weight_decay_results.items())):
    history = data['history']
    epochs_range = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    ax_loss = axes[i, 0]
    ax_loss.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    ax_loss.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
    ax_loss.set_title(f'Seed {seed} (Weight Decay 1e-3): Loss vs Epochs')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)

    # Plot Accuracy
    ax_acc = axes[i, 1]
    ax_acc.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
    ax_acc.plot(epochs_range, history['val_acc'], label='Val Acc', marker='o')
    ax_acc.set_title(f'Seed {seed} (Weight Decay 1e-3): Accuracy vs Epochs')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)

    # Collect data for summary
    wd_summary_data.append({
        'Seed': seed,
        'Test Accuracy': data['test_acc'],
        'Epochs to Converge': data['epochs'],
        'Time (s)': data['time']
    })
    wd_test_accuracies.append(data['test_acc'])

plt.tight_layout()
plt.show()

# 3. Create Summary DataFrame
weight_decay_summary_df = pd.DataFrame(wd_summary_data)

# 4. Calculate Statistics
wd_mean_acc = np.mean(wd_test_accuracies)
wd_std_acc = np.std(wd_test_accuracies)

# 5. Print Report
print("\n--- Weight Decay (L2 = 1e-3) Performance Summary ---")
print(weight_decay_summary_df.to_string(index=False))
print("\n--- Aggregate Statistics ---")
print(f"Mean Test Accuracy: {wd_mean_acc:.4f}")
print(f"Std Dev Test Accuracy: {wd_std_acc:.4f}")


# Save Stage 3 (combined) Results to ONE CSV
rows = []

# Dropout rows 
for row in dropout_summary_data:
    rows.append({
        "Method": "Dropout",
        "Seed": row["Seed"],
        "Test Accuracy": row["Test Accuracy"],
        "Epochs to Converge": row["Epochs to Converge"],
        "Time (s)": row["Time (s)"],
        "lr": 0.01,
        "momentum": 0.9,
        "init_type": "xavier",
        "activation": activation,
        "dropout_rate": dropout_rate,     
        "weight_decay": 0.0,
        "wd_mean_acc": wd_mean_acc,
        "wd_std_acc": wd_std_acc,
        "dropout_mean_acc": dropout_mean_acc
    })

# --- Weight decay rows ---
for row in wd_summary_data:
    rows.append({
        "Method": "Weight Decay",
        "Seed": row["Seed"],
        "Test Accuracy": row["Test Accuracy"],
        "Epochs to Converge": row["Epochs to Converge"],
        "Time (s)": row["Time (s)"],
        "lr": 0.01,
        "momentum": 0.9,
        "init_type": "xavier",
        "activation": activation,
        "dropout_rate": 0.0,
        "weight_decay": weight_decay      
    })

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("results/regularization.csv", index=False)

