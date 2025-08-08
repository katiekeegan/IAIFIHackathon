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

def init_fanin(m, nonlinearity="relu"):
    # Works well for ReLU/GELU networks
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)

# (Optional) If you want the very last classifier layer to start small:
def zero_last_linear_weights(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)