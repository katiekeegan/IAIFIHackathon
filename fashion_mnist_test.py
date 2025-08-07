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


