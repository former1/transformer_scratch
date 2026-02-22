import numpy as np
import math
import torch 
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# classical scaled dot product attention
def scaled_dotprod(q,k,v,mask=None):
    d_k = k.size(-1)
    scaled= torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(d_k) # includes necessary transposing on k, regarding batch size and nheads
    if mask is not None:
        scaled = scaled.permute(1,0,2,3) + mask
    attention= F.softmax(scaled, dim=-1)

# implementation of the sin-cos positional encoding
# Mainly helps us take the positional awareness in text, so that we can make the distinction between "man bites dog" and "dog bites man".
class positional_encoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super.__init__()
        self.max_sequence_length= max_sequence_length
        self.d_model= d_model
        def forward(self):
            pos= torch.arange(self.max_sequence_length).unsqueeze(1)
            divident= torch.exp(torch.arange(0,self.d_model,2)* -(math.log(10000.0)/self.d_model))
            
            pe= torch.zeros(self.max_sequence_length, self.d_model)
            pe[:, 0::2]= torch.sin(pos*divident)
            pe[:, 1::2]= torch.cos(pos*divident)
            return pe
        
