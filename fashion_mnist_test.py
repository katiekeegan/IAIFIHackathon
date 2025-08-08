import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from utils import *
from models import *

torch.backends.cudnn.benchmark = True

# -------------------
# Setup
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

def train_one(model, opt, criterion, loader, epochs=2, val_loader=None):
    best_mse = float("inf")
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
        # monitor after each epoch
        if val_loader is not None:
            curr_mse = eval_mse(model, val_loader)
            best_mse = min(best_mse, curr_mse)
    return best_mse

@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.numel()
    return correct / total

@torch.no_grad()
def eval_mse(model, loader):
    model.eval()
    mse_sum, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        y_oh = torch.nn.functional.one_hot(yb, num_classes=probs.shape[1]).float()
        # per-sample MSE over classes, then average over batch
        batch_mse = ((probs - y_oh)**2).mean(dim=1)
        mse_sum += batch_mse.sum().item()
        n += yb.size(0)
    return mse_sum / n

def run_trial(num_layers, lr, alpha=1.0, epochs=2, hidden_dim=512, weight_decay=0.0, adam_eps=1e-8):
    model = ResidualMLPCompleteP(
        input_dim=28*28, hidden_dim=hidden_dim, num_layers=num_layers,
        num_classes=10, alpha=alpha
    ).to(device)

    model.apply(lambda m: init_fanin(m, nonlinearity="relu"))

    criterion = nn.CrossEntropyLoss()
    param_groups = make_param_groups(model, base_lr=lr, L=num_layers, alpha=alpha)
    opt = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, eps=adam_eps)

    best_mse = train_one(model, opt, criterion, train_loader, epochs=epochs, val_loader=test_loader)
    acc = eval_acc(model, test_loader)
    mse = eval_mse(model, test_loader)

    return acc, mse, best_mse


# Folder for results
save_dir = "fashion_mnist"
os.makedirs(save_dir, exist_ok=True)

# -------------------
# Data
# -------------------
transform = transforms.ToTensor()
train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

subset_size = 15000
train_indices = np.random.RandomState(0).choice(len(train_ds), size=subset_size, replace=False)
train_sub = Subset(train_ds, train_indices)

train_loader = DataLoader(train_sub, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=1, pin_memory=True)

# -------------------
# Sweep settings
# -------------------
depths = [2, 4, 8, 16, 32, 64, 128]
lrs = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
EPOCHS = 2
H = 512
alphas = [0.5, 1.0]  # <-- change as needed

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
from time import time

def run_one(depth, lr, alpha):
    acc = run_trial(
        num_layers=depth,
        lr=float(lr),
        alpha=alpha,
        epochs=EPOCHS,
        hidden_dim=H,
        weight_decay=0.0
    )
    return {"num_layers": depth, "alpha": alpha, "lr": float(lr), "accuracy": acc}

# -------------------
# Main loop
# -------------------
for alpha in alphas:
    print(f"\n=== Running alpha={alpha} ===")
    records = []
    t0 = time()
    for d in depths:
        for lr in lrs:
            acc, mse, best_mse = run_trial(
                num_layers=d, lr=float(lr), alpha=alpha,
                epochs=EPOCHS, hidden_dim=H, weight_decay=0.0
            )
            records.append({"num_layers": d, "alpha": alpha, "lr": float(lr),
                            "accuracy": acc, "mse": mse, "best_mse": best_mse})
            print(f"Depth {d} | LR {lr:.1e} | Acc: {acc:.4f} | MSE: {mse:.6f} | Best MSE: {best_mse:.6f}")

    elapsed = time() - t0
    df = pd.DataFrame(records).sort_values(["num_layers", "lr"])

    # Save CSV
    csv_path = os.path.join(save_dir, f"fashion_alpha_{alpha}.csv")
    df.to_csv(csv_path, index=False)

    # --- Accuracy plot (unchanged) ---
    plt.figure(figsize=(7, 5))
    for d in depths:
        sub = df[df["num_layers"] == d].sort_values("lr")
        plt.plot(sub["lr"], sub["accuracy"], marker='o', label=f"{d} layers")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.title(f"Fashion-MNIST (alpha={alpha}): Accuracy vs Learning Rate")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fashion_alpha_{alpha}_accuracy.png"), dpi=300)
    plt.close()

    # --- MSE plot ---
    plt.figure(figsize=(7, 5))
    min_mse_val = df["mse"].min()
    for d in depths:
        sub = df[df["num_layers"] == d].sort_values("lr")
        plt.plot(sub["lr"], sub["mse"], marker='o', label=f"{d} layers")
    plt.axhline(min_mse_val, color='red', linestyle='--', linewidth=1,
                label=f"Lowest MSE = {min_mse_val:.6f}")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test MSE")
    plt.title(f"Fashion-MNIST (alpha={alpha}): MSE vs Learning Rate")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fashion_alpha_{alpha}_mse.png"), dpi=300)
    plt.close()