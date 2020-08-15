import string
import re
import random
import time

import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torchtext


def init_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
