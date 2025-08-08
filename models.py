import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# --- Reproducibility ---
torch.manual_seed(0)
np.random.seed(0)

# --- Scaling MLP ---
class ScalingMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=3,
                 activation=nn.ReLU, batch_norm=True):
        super().__init__()
        layers = []
        if num_layers == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(activation())
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ----- Model -----

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, alpha, L):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
            nn.Linear(dim, dim),
        )
        self.alpha = alpha
        self.L = max(L, 1)  # avoid divide by zero when L=0

    def forward(self, x):
        # Pre-LN residual: h -> h + L^{-alpha} * FF(LN(h))
        return x + (self.L ** (-self.alpha)) * self.ff(self.ln(x))

class ResidualMLPCompleteP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=512, num_layers=4, num_classes=10, alpha=1.0):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden_dim) if num_layers > 0 else None
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, alpha=alpha, L=num_layers)
                                     for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(hidden_dim) if num_layers > 0 else None
        self.out = nn.Linear(hidden_dim if num_layers > 0 else input_dim, num_classes)
        self.num_layers = num_layers
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.num_layers == 0:
            return self.out(x)  # direct linear model
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_ln(h)
        return self.out(h)
