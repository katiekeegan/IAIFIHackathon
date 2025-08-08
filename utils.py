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
