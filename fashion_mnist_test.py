import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

from utils import *
from models import *

"""
Brief scaling experiments on FashionMNIST dataset with architecture based on Figure 5 in https://arxiv.org/pdf/2505.01618.
"""

# ----- Data -----
transform = transforms.ToTensor()  # [0,1]
train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.FashionMNIST(root="./data",  train=False, download=True, transform=transform)

# Light subset for speed (e.g., 15000 examples)
subset_size = 15000
train_indices = np.random.RandomState(0).choice(len(train_ds), size=subset_size, replace=False)
train_sub = Subset(train_ds, train_indices)

train_loader = DataLoader(train_sub, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=1, pin_memory=True)


# ----- Sweep settings -----
depths = [2, 4, 8, 16]           # lines
lrs = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])  # x-axis
# NOTE: more small learning rates
EPOCHS = 2
H = 512
ALPHA = 1.0                  # CompleteP by default; change to 0.5 to compare

records = []
t0 = time()
for d in depths:
    for lr in lrs:
        acc = run_trial(num_layers=d, lr=float(lr), alpha=ALPHA, epochs=EPOCHS, hidden_dim=H, weight_decay=0.0, dropout=0.0)
        records.append({"num_layers": d, "alpha": ALPHA, "lr": float(lr), "accuracy": acc})
        print(f"Experiment with {d} layers with learning rate {lr} has been done! Accuracy: {acc}")
elapsed = time() - t0

df = pd.DataFrame(records).sort_values(["num_layers", "lr"])

# Plot: accuracy vs learning rate (log x), lines per depth
plt.figure(figsize=(7,5))
for d in depths:
    sub = df[df["num_layers"] == d].sort_values("lr")
    plt.plot(sub["lr"], sub["accuracy"], marker='o', label=f"{d} layers")
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Test Accuracy")
plt.title(f"Fashion-MNIST (alpha={ALPHA}): Accuracy vs Learning Rate (lines = depth)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

df.head(), f"Elapsed ~{elapsed:.1f}s"