import numpy as np
import math
import torch 
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

