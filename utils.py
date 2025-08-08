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
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

def make_param_groups(model, base_lr, L, alpha):
    """
    Create optimizer param groups with LR scaled by L^(alpha-1) for hidden/LN/bias params,
    following the depth rule in Table 1 (ignoring width scaling for simplicity).
    """
    scale = (L ** (alpha - 1.0)) if L > 0 else 1.0
    groups = []

    # Separate output head, input layer, and hidden/LN/bias
    hidden_params, head_params, in_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "out" in name:
            head_params.append(p)
        elif "inp" in name or "blocks" in name or "final_ln" in name:
            hidden_params.append(p)
        else:
            in_params.append(p)

    # Hidden / LN / bias group with scaled LR
    if hidden_params:
        groups.append({"params": hidden_params, "lr": base_lr * scale})
    # Input & head groups at base LR (emb/unemb in Table 1 keep base η w.r.t. depth)
    if in_params:
        groups.append({"params": in_params, "lr": base_lr})
    if head_params:
        groups.append({"params": head_params, "lr": base_lr})

    return groups

# ----- Train / Eval -----
def train_one(model, opt, criterion, loader, epochs=2):
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()


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

def run_trial(num_layers, lr, alpha=1.0, epochs=2, hidden_dim=512, weight_decay=0.0, dropout=0.0, adam_eps=1e-8):
    model = ResidualMLPCompleteP(
        input_dim=28*28, hidden_dim=hidden_dim, num_layers=num_layers,
        num_classes=10, alpha=alpha, dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # AdamW groups: apply L^(alpha-1) to hidden/LN/bias group, base LR elsewhere
    param_groups = make_param_groups(model, base_lr=lr, L=num_layers, alpha=alpha)
    opt = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, eps=adam_eps)

    train_one(model, opt, criterion, train_loader, epochs=epochs)
    acc = eval_acc(model, test_loader)
    return acc

