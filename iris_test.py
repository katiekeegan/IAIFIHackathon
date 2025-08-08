# Disable Dynamo to avoid environment-specific import issues
import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, load_f
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from utils import *
from models import *

# --- Reproducibility ---
torch.manual_seed(0)
np.random.seed(0)

# === Prepare Iris dataset ===
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# --- Training/Eval ---
def train_and_eval(num_layers, lr, epochs=100, hidden_dim=64, weight_decay=0.0):
    model = ScalingMLP(
        input_dim=4, output_dim=3,
        hidden_dim=hidden_dim, num_layers=num_layers,
        activation=nn.ReLU, batch_norm=True
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()

    print(f"Experiment with {num_layers} layers with learning rate {lr} has been done! Accuracy: {acc}")
    return acc

import os
os.makedirs("/mnt/data", exist_ok=True)

# --- Sweep settings ---
depths = [1, 2, 4, 8, 16, 32, 64]
lrs = np.array([1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 3.0, 10.0])

records = []
for d in depths:
    for lr in lrs:
        acc = train_and_eval(num_layers=d, lr=float(lr), epochs=120, hidden_dim=64, weight_decay=0.0)
        records.append({"num_layers": d, "lr": float(lr), "accuracy": acc})

df = pd.DataFrame(records)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"/mnt/data/iris_scaling_lr_depth_{timestamp}.csv"
df.to_csv(csv_path, index=False)

# # Display table to user
# display_dataframe_to_user("Iris scaling results (accuracy vs learning rate, lines = num_layers)", df)

# --- Plot: accuracy vs learning rate (log scale), lines per depth ---
plt.figure(figsize=(7,5))
for d in depths:
    sub = df[df["num_layers"] == d].sort_values("lr")
    # plot on a log-x
    plt.plot(sub["lr"], sub["accuracy"], marker='o', label=f"{d} layers")
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Test Accuracy")
plt.title("Iris: Accuracy vs Learning Rate (lines = depth)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plot_path = f"/mnt/data/iris_scaling_lr_depth_plot_{timestamp}.png"
plt.tight_layout()
plt.savefig(plot_path)
plt.show()
