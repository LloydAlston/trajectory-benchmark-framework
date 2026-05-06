import torch
import random
import numpy as np
import yaml

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)