import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from utils import *
from models import *

# -------------------
# Setup
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

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
depths = [2, 4, 8, 16]
lrs = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
EPOCHS = 2
H = 512
alphas = [0.5, 1.0, 2.0]  # <-- change as needed

# -------------------
# Main loop
# -------------------
for alpha in alphas:
    print(f"\n=== Running alpha={alpha} ===")
    records = []
    t0 = time()
    for d in depths:
        for lr in lrs:
            acc = run_trial(
                num_layers=d,
                lr=float(lr),
                alpha=alpha,
                epochs=EPOCHS,
                hidden_dim=H,
                weight_decay=0.0,
                dropout=0.0
            )
            records.append({"num_layers": d, "alpha": alpha, "lr": float(lr), "accuracy": acc})
            print(f"Depth {d} | LR {lr:.1e} | Accuracy: {acc:.4f}")

    elapsed = time() - t0
    df = pd.DataFrame(records).sort_values(["num_layers", "lr"])

    # Save CSV
    csv_path = os.path.join(save_dir, f"fashion_alpha_{alpha}.csv")
    df.to_csv(csv_path, index=False)

    # Plot
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

    # Save figure
    fig_path = os.path.join(save_dir, f"fashion_alpha_{alpha}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved results for alpha={alpha} in {save_dir} (elapsed {elapsed:.1f}s)")