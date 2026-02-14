# stage5.py
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ---- Make src importable ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

from src.common import (
    load_stage1_artifacts,
    make_loaders,
    initialize_weights,
    EarlyStopping,
    CIFAR10Subset,
    build_transform,
)

# Paths
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUPERVISED_CSV = os.path.join(RESULTS_DIR, "supervised.csv")
REGULARIZATION_CSV = os.path.join(RESULTS_DIR, "regularization.csv")
SEMISUP_CSV = os.path.join(RESULTS_DIR, "semi-supervised.csv")

OUT_COMPONENT_CSV = os.path.join(RESULTS_DIR, "stage5_component_analysis.csv")
OUT_LABELPCT_CSV = os.path.join(RESULTS_DIR, "stage5_varying_labels.csv")
OUT_COST_CSV = os.path.join(RESULTS_DIR, "stage5_cost_analysis.csv")

# Load Stage 1 artifacts + loaders
loaded, mean, std = load_stage1_artifacts()

# make_loaders returns:
# labeled_dataset, val_dataset, test_dataset, labeled_loader, val_loader, test_loader, full_data, unlabeled_indices
labeled_ds, val_ds, test_ds, labeled_loader, val_loader, test_loader, full_data, unlabeled_indices = make_loaders(batch_size=64)
transform = build_transform(mean, std)

# Needed for Stage 5b/5c
labeled_indices = loaded["labeled_indices"]
y_labeled = loaded["y_labeled"]
y_unlabeled = loaded["y_unlabeled"]  

# Load Results CSVs (Stages 2–4)
supervised_df = pd.read_csv(SUPERVISED_CSV)
regularization_df = pd.read_csv(REGULARIZATION_CSV)
semisup_df = pd.read_csv(SEMISUP_CSV)

# --- Stage 2 baseline stats ---
baseline_accs = supervised_df["Test Accuracy"].astype(float).to_numpy()
baseline_mean_acc = float(np.mean(baseline_accs))
baseline_std_acc = float(np.std(baseline_accs, ddof=0))

baseline_times = supervised_df["Time (s)"].astype(float).to_numpy()
baseline_mean_time = float(np.mean(baseline_times))

# --- Stage 3 stats: Dropout + Weight Decay ---
def method_stats(df: pd.DataFrame, method_name: str):
    sub = df[df["Method"].astype(str).str.lower() == method_name.lower()].copy()
    if len(sub) == 0:
        return 0.0, 0.0, 0.0, np.array([])
    accs = sub["Test Accuracy"].astype(float).to_numpy()
    times = sub["Time (s)"].astype(float).to_numpy()
    return float(np.mean(accs)), float(np.std(accs, ddof=0)), float(np.mean(times)), accs

dropout_mean_acc, dropout_std_acc, dropout_mean_time, dropout_accs = method_stats(regularization_df, "Dropout")
wd_mean_acc, wd_std_acc, wd_mean_time, wd_accs = method_stats(regularization_df, "Weight Decay")

# --- Stage 4 stats: Consistency Regularization ---
consistency_accs = semisup_df["test_accuracy"].astype(float).to_numpy()
consistency_mean_acc = float(np.mean(consistency_accs))
consistency_std_acc = float(np.std(consistency_accs, ddof=0))

consistency_times = semisup_df["training_time_seconds"].astype(float).to_numpy()
consistency_mean_time = float(np.mean(consistency_times))

# Stage 5a: Component Contribution Analysis (from saved CSVs)
component_analysis_data = [
    {"Model": "Baseline (10% Labels)", "Mean Test Accuracy": baseline_mean_acc, "Std Dev": baseline_std_acc},
]

if dropout_mean_acc > 0:
    component_analysis_data.append(
        {"Model": "Dropout (0.5)", "Mean Test Accuracy": dropout_mean_acc, "Std Dev": dropout_std_acc}
    )

if wd_mean_acc > 0:
    component_analysis_data.append(
        {"Model": "Weight Decay (1e-3)", "Mean Test Accuracy": wd_mean_acc, "Std Dev": wd_std_acc}
    )

component_analysis_data.append(
    {"Model": "Consistency Reg. (Stage 4)", "Mean Test Accuracy": consistency_mean_acc, "Std Dev": consistency_std_acc}
)

component_df = pd.DataFrame(component_analysis_data)
component_df.to_csv(OUT_COMPONENT_CSV, index=False)

print("\n=== Stage 5a: Component Contribution Analysis ===")
print(component_df.to_string(index=False))
print(f"\nSaved -> {OUT_COMPONENT_CSV}")

# Stage 5b: Fully Supervised Upper Bound (100% labels)
full_train_indices = np.concatenate([labeled_indices, unlabeled_indices])
full_train_labels = np.concatenate([y_labeled, y_unlabeled])

full_train_ds = CIFAR10Subset(full_data, full_train_indices, full_train_labels, transform=transform)
full_train_loader = DataLoader(full_train_ds, batch_size=64, shuffle=True)

class FeedForwardNN(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=(256, 128), output_size=10,
                 activation="relu", dropout_rate=0.0):
        super().__init__()
        layers = []

        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "sigmoid":
            act_fn = nn.Sigmoid()
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        cur = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(cur, h))
            layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            cur = h

        layers.append(nn.Linear(cur, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.net(x)

def run_fully_supervised(seed: int,
                         max_epochs=50,
                         lr=0.01,
                         momentum=0.9,
                         weight_decay=1e-3,
                         init_type="xavier",
                         activation="relu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForwardNN(activation=activation).to(device)
    initialize_weights(model, init_type=init_type, activation=activation)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    early = EarlyStopping(window_size=5)

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in full_train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # validation loss for early stopping
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_n += xb.size(0)

        if val_n > 0 and early.check_stop(val_loss_sum / val_n):
            break

    # test accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0

fs_seeds = [1, 123, 12345]
fs_accs = [run_fully_supervised(s) for s in fs_seeds]
fs_mean = float(np.mean(fs_accs))
fs_std = float(np.std(fs_accs, ddof=0))

print("\n=== Stage 5b: Fully Supervised Upper Bound (100% Labels) ===")
print(f"Accuracies per seed: {fs_accs}")
print(f"Mean: {fs_mean:.4f} | Std: {fs_std:.4f}")

# Add upper bound to component table (optional but useful)
component_df2 = pd.concat(
    [component_df, pd.DataFrame([{"Model": "Upper Bound (100% Labels)", "Mean Test Accuracy": fs_mean, "Std Dev": fs_std}])],
    ignore_index=True
)
component_df2.to_csv(OUT_COMPONENT_CSV, index=False)

# Stage 5c: Varying labeled percentage (semi-supervised consistency)
consistency_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
])

class CIFAR10UnlabeledSubset(Dataset):
    """
    Returns (raw_numpy_image, augmented_tensor_image)
    raw_numpy_image is HWC uint8, used to build x (normalized only) in the loop.
    """
    def __init__(self, full_data, indices, transform=None):
        self.full_data = full_data
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        original_idx = self.indices[i]
        img = self.full_data[original_idx]  # HWC uint8 numpy
        x_prime = self.transform(img) if self.transform else torch.from_numpy(img).permute(2,0,1).float() / 255.0
        return img, x_prime

def get_loaders_for_percentage(p: float, seed: int, full_indices, full_labels):
    np.random.seed(seed)
    n = len(full_indices)
    n_l = int(p * n)

    perm = np.random.permutation(n)
    idx_shuf = full_indices[perm]
    y_shuf = full_labels[perm]

    idx_l = idx_shuf[:n_l]
    y_l = y_shuf[:n_l]
    idx_ul = idx_shuf[n_l:]

    ds_l = CIFAR10Subset(full_data, idx_l, y_l, transform=transform)
    ds_ul = CIFAR10UnlabeledSubset(full_data, idx_ul, transform=consistency_transform)

    l_loader = DataLoader(ds_l, batch_size=64, shuffle=True)
    ul_loader = DataLoader(ds_ul, batch_size=64, shuffle=True)
    return l_loader, ul_loader

def run_consistency_with_loaders(seed: int, l_loader, ul_loader,
                                 max_epochs=50,
                                 lr=0.01,
                                 momentum=0.9,
                                 weight_decay=0.0,
                                 lambda_consistency=1.0,
                                 warmup_epochs=10,
                                 init_type="xavier",
                                 activation="relu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForwardNN(activation=activation).to(device)
    initialize_weights(model, init_type=init_type, activation=activation)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    early = EarlyStopping(window_size=5)

    normalize_only = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    for epoch in range(max_epochs):
        model.train()
        l_it = iter(l_loader)
        ul_it = iter(ul_loader)
        num_batches = len(l_loader)

        for _ in range(num_batches):
            try:
                x_l, y_l = next(l_it)
            except StopIteration:
                l_it = iter(l_loader)
                x_l, y_l = next(l_it)

            try:
                x_ul_raw, x_ul_prime = next(ul_it)
            except StopIteration:
                ul_it = iter(ul_loader)
                x_ul_raw, x_ul_prime = next(ul_it)

            x_l, y_l = x_l.to(device), y_l.to(device)

            # raw unlabeled: uint8 HWC -> float BCHW -> normalize
            x_ul = x_ul_raw.permute(0, 3, 1, 2).float() / 255.0
            x_ul = normalize_only(x_ul).to(device)
            x_ul_prime = x_ul_prime.to(device)

            optimizer.zero_grad()

            # supervised
            out_l = model(x_l)
            loss_sup = criterion(out_l, y_l)

            # consistency (teacher = model(x_ul) stopgrad)
            with torch.no_grad():
                p_teacher = torch.softmax(model(x_ul), dim=1)

            p_student = torch.softmax(model(x_ul_prime), dim=1)

            # MSE between distributions (your later stage5 version)
            loss_cons = F.mse_loss(p_student, p_teacher)

            cur_lambda = lambda_consistency * min((epoch + 1) / warmup_epochs, 1.0) if warmup_epochs > 0 else lambda_consistency
            loss = loss_sup + cur_lambda * loss_cons

            loss.backward()
            optimizer.step()

        # validation for early stopping
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                l = criterion(out, yb)
                val_loss_sum += l.item() * xb.size(0)
                val_n += xb.size(0)

        if val_n > 0 and early.check_stop(val_loss_sum / val_n):
            break

    # test acc
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0

percentages = [0.05, 0.10, 0.25, 0.50]
seeds = [1, 123, 12345]

vary_rows = []
print("\n=== Stage 5c: Varying labeled percentage (Consistency Reg.) ===")
for p in percentages:
    accs = []
    for s in seeds:
        l_loader, ul_loader = get_loaders_for_percentage(p, s, full_train_indices, full_train_labels)
        acc = run_consistency_with_loaders(s, l_loader, ul_loader)
        accs.append(acc)
        print(f"Label%={int(p*100):>2}% | seed={s:<5} acc={acc:.4f}")
    vary_rows.append({
        "label_percent": int(p * 100),
        "mean_test_accuracy": float(np.mean(accs)),
        "std_test_accuracy": float(np.std(accs, ddof=0)),
        "seed1": accs[0],
        "seed123": accs[1],
        "seed12345": accs[2],
    })

vary_df = pd.DataFrame(vary_rows)
vary_df.to_csv(OUT_LABELPCT_CSV, index=False)
print(f"\nSaved -> {OUT_LABELPCT_CSV}")

# Plot
plt.figure(figsize=(9, 6))
x = vary_df["label_percent"].to_numpy()
y = vary_df["mean_test_accuracy"].to_numpy()
yerr = vary_df["std_test_accuracy"].to_numpy()
plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=5)
plt.xlabel("Percentage of Labeled Data (%)")
plt.ylabel("Test Accuracy")
plt.title("Stage 5c: Test Accuracy vs Labeled Data % (Consistency Regularization)")
plt.grid(True)
plt.xticks(x)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage5_varying_labels_plot.png"), dpi=200)
plt.show()

# Stage 5d: Statistical significance (paired t-test baseline vs consistency)
try:
    from scipy import stats
    # Align by seed order
    # supervised.csv uses Seed column, semi-supervised.csv uses seed column
    baseline_by_seed = supervised_df.set_index("Seed")["Test Accuracy"].astype(float)
    consistency_by_seed = semisup_df.set_index("seed")["test_accuracy"].astype(float)

    seeds_common = [1, 123, 12345]
    b = np.array([baseline_by_seed.loc[s] for s in seeds_common], dtype=float)
    c = np.array([consistency_by_seed.loc[s] for s in seeds_common], dtype=float)

    t_stat, p_val = stats.ttest_rel(c, b)

    print("\n=== Stage 5d: Paired t-test (Consistency vs Baseline) ===")
    print(f"Baseline accs:    {b.tolist()}")
    print(f"Consistency accs: {c.tolist()}")
    print(f"t = {t_stat:.4f}, p = {p_val:.6f}")
    print("Significant at p<0.05 ?" , "YES" if p_val < 0.05 else "NO")
except Exception as e:
    print("\n=== Stage 5d: Paired t-test skipped ===")
    print("Reason:", str(e))

# Stage 5e: Computational cost vs gain
cost_rows = [
    {
        "Method": "Baseline (Stage 2)",
        "Mean Time (s)": baseline_mean_time,
        "Mean Acc": baseline_mean_acc,
        "Acc Gain vs Baseline": 0.0,
    },
]

if dropout_mean_time > 0:
    cost_rows.append({
        "Method": "Dropout (Stage 3)",
        "Mean Time (s)": dropout_mean_time,
        "Mean Acc": dropout_mean_acc,
        "Acc Gain vs Baseline": dropout_mean_acc - baseline_mean_acc,
    })

if wd_mean_time > 0:
    cost_rows.append({
        "Method": "Weight Decay (Stage 3)",
        "Mean Time (s)": wd_mean_time,
        "Mean Acc": wd_mean_acc,
        "Acc Gain vs Baseline": wd_mean_acc - baseline_mean_acc,
    })

cost_rows.append({
    "Method": "Consistency Reg. (Stage 4)",
    "Mean Time (s)": consistency_mean_time,
    "Mean Acc": consistency_mean_acc,
    "Acc Gain vs Baseline": consistency_mean_acc - baseline_mean_acc,
})

cost_df = pd.DataFrame(cost_rows)
cost_df.to_csv(OUT_COST_CSV, index=False)

print("\n=== Stage 5e: Computational Cost vs Gain ===")
print(cost_df.to_string(index=False))
print(f"\nSaved -> {OUT_COST_CSV}")

# Optional: quick tradeoff print
if baseline_mean_time > 0:
    ratio = consistency_mean_time / baseline_mean_time
    gain = (consistency_mean_acc - baseline_mean_acc) * 100
    print(f"\nTrade-off: Consistency takes ~{ratio:.2f}x baseline time, for +{gain:.2f}% test accuracy.")

