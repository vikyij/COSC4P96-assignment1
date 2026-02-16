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

## Define Network Architecture - Implement the `FeedForwardNN` class with configurable layers and activations.
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
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
            current_dim = hidden_dim

        # Output layer (no activation, returns logits for CrossEntropyLoss)
        layers.append(nn.Linear(current_dim, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input: [batch, C, H, W] -> [batch, input_size]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.network(x)

# Instantiate a test model to verify architecture
# CIFAR-10 images: 32x32x3 = 3072 features
input_size = 3072
hidden_sizes = [256, 128]
output_size = 10
activation = 'relu'

model = FeedForwardNN(input_size, hidden_sizes, output_size, activation)
print("Model Architecture:")
print(model)


print("Training utilities (initialize_weights, EarlyStopping) implemented.")

"""## Run Baseline Experiments
Execute the training loop using SGD with Momentum on the labeled dataset for 3 different random seeds (1, 123, 12345). Track training time, training/validation loss, and accuracy.
"""

# Added init_type parameter to allow comparison of different initialization methods
def run_experiment(seed, input_size=3072, hidden_sizes=[256, 128], output_size=10,
                   activation='relu', lr=0.01, momentum=0.9, max_epochs=50, init_type='xavier'):
    # a. Set random seeds (Requirement: Run experiments with different random seeds)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running experiment with seed {seed}, init {init_type} on {device}...")

    # b. Instantiate Model
    model = FeedForwardNN(input_size, hidden_sizes, output_size, activation).to(device)

    # c. Initialize Weights (Requirement: Implement and compare different initializations)
    # Now uses the passed init_type argument ('uniform', 'normal', or 'xavier')
    initialize_weights(model, init_type=init_type, activation=activation)

    # d. Define Optimizer and Loss
    # Requirement: Backpropagation with cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Requirement: Optimizer SGD, Mini-batch (via loader), Momentum, Learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # e. Instantiate Early Stopping
    # Requirement: Implement early stopping with specific criterion (EV > mean + std)
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

        # Requirement: Train using only the 20% labeled data
        for inputs, labels in labeled_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # Backpropagation
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

# 4. Loop through seeds and run experiments
seeds = [1, 123, 12345]
results = {}

print("Starting Baseline Experiments...\n")

for seed in seeds:
    print(f"--- Experiment with Seed {seed} ---")
    # Using default init_type='xavier' as the baseline
    hist, time_taken, epochs, test_acc = run_experiment(seed, init_type='xavier')
    results[seed] = {
        'history': hist,
        'time': time_taken,
        'epochs': epochs,
        'test_acc': test_acc
    }
    print(f"Seed {seed} Completed: Test Acc: {test_acc:.4f}, Epochs: {epochs}, Time: {time_taken:.2f}s\n")

print("All experiments completed.")


## Visualize and Report Results
# 1. Setup the figure for plotting
num_seeds = len(results)
fig, axes = plt.subplots(num_seeds, 2, figsize=(15, 5 * num_seeds))

# Ensure axes is 2D array even if num_seeds is 1
if num_seeds == 1:
    axes = np.array([axes])

summary_data = []
test_accuracies = []

# 2. Iterate through results and plot
for i, (seed, data) in enumerate(sorted(results.items())):
    history = data['history']
    epochs_range = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    ax_loss = axes[i, 0]
    ax_loss.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    ax_loss.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
    ax_loss.set_title(f'Seed {seed}: Loss vs Epochs')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)

    # Plot Accuracy
    ax_acc = axes[i, 1]
    ax_acc.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
    ax_acc.plot(epochs_range, history['val_acc'], label='Val Acc', marker='o')
    ax_acc.set_title(f'Seed {seed}: Accuracy vs Epochs')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)

    # Collect data for summary
    summary_data.append({
        'Seed': seed,
        'Test Accuracy': data['test_acc'],
        'Epochs to Converge': data['epochs'],
        'Time (s)': data['time']
    })
    test_accuracies.append(data['test_acc'])

plt.tight_layout()
plt.show()

# 3. Create Summary DataFrame
summary_df = pd.DataFrame(summary_data)

# 4. Calculate Statistics
mean_acc = np.mean(test_accuracies)
std_acc = np.std(test_accuracies)

# 5. Print Report
print("\n--- Performance Summary ---")
print(summary_df.to_string(index=False))
print("\n--- Aggregate Statistics ---")
print(f"Mean Test Accuracy: {mean_acc:.4f}")
print(f"Std Dev Test Accuracy: {std_acc:.4f}")

# Save Results to CSV

# Add hyperparameters to summary
for row in summary_data:
    row.update({
        "lr": 0.01,
        "momentum": 0.9,
        "init_type": "xavier",
        "activation": "relu",
        "mean_test_accuracy": float(mean_acc),
        "std_test_accuracy": float(std_acc)
    })

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(summary_data)
df.to_csv("results/supervised.csv", index=False)