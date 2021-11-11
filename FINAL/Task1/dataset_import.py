import numpy as np
import os
import pickle
import torch

from torch.utils.data import Dataset, DataLoader

CATEGORY_INDEX = {
    "negative": 0,
    "positive": 1
}

