import random

import numpy as np
import torch


DTYPE = torch.float32


def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_tensor(value, device, dtype=DTYPE):
    return torch.as_tensor(value, dtype=dtype, device=device)
