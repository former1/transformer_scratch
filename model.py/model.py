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
        super().__init__()
        self.max_sequence_length= max_sequence_length
        self.d_model= d_model
        def forward(self):
            pos= torch.arange(self.max_sequence_length).unsqueeze(1)
            divident= torch.exp(torch.arange(0,self.d_model,2)* -(math.log(10000.0)/self.d_model))
            
            pe= torch.zeros(self.max_sequence_length, self.d_model)
            pe[:, 0::2]= torch.sin(pos*divident)
            pe[:, 1::2]= torch.cos(pos*divident)
            return pe



# input = sentence with max length of max_sequence_length, 
# output  = embedded sentence with positional encoding and dropout, ready to be fed into the encoder layers.
# given a sentence, we first convert it to token ids, then apply embedding with dim = d_model
# afterwards we add positional encoding and apply dropout.
class sentence_embedding(x):

    def __init__(self,  max_sequence_length,d_model, language_index_convert, START_TOKEN, END_TOKEN, PAD_TOKEN):
        super().__init__()
        self.vocab_size= len(language_index_convert)
        self.max_sequence_length= max_sequence_length
        self.language_index_convert= language_index_convert
        pos_encoder= positional_encoding(d_model, max_sequence_length)
        self.dropout= nn.Dropout(0.1)
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.START_TOKEN= START_TOKEN
        self.END_TOKEN= END_TOKEN
        self.PAD_TOKEN= PAD_TOKEN        

# classical mha, we use the same linear layer to project the input into q,k,v.
# Then we split the projected output into q,k,v
class multi_head_attention(q,k,v,mask=None):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model= d_model
        self.n_heads= n_heads
        self.dim_head= d_model//n_heads
        self.qkv_nn_layer= nn.Linear(d_model, 3*d_model)
        self.final_linear_layer= nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_length, d_model= x.size()
        qkv= self.qkv_nn_layer(x)
        qkv= qkv.reshape(batch_size, seq_length, self.n_heads, 3*self.dim_head)
        qkv= qkv.permute(0,2,1,3)
        q, k, v= qkv.chunk(3, dim=-1)
        values, attention= scaled_dotprod(q,k,v,mask)
        values= values.permute(0,2,1,3).reshape(batch_size, seq_length, self.n_heads*self.dim_head)
        out=self.final_linear_layer(values)
        return out
        
       
# we are normalizing across the features,
# so we need to specify the dimensions of the features in the inputt
class layer_normalization(nn.Module):
    
    def __init__(self,parameters, eps=1e-6):
        super().__init__()
        self.parameters= parameters
        self.eps= eps
        self.gamma= nn.Parameter(torch.ones(parameters))
        self.beta= nn.Parameter(torch.zeros(parameters))

    def forward(self, x):
        dims=  [-(i+1)  for i in range(len(self.parameters))]
        mean= x.mean(dim=dims, keepdim=True)
        var =((x-mean)**2).mean(dim=dims, keepdim=True)
        stdev= (var + self.eps).sqrt()
        y= (x-mean)/stdev
        out= self.gamma*y + self.beta
        return out

class feed_forward_network(nn.Module):
    def __init__(self, d_model, ff_hidden_dim, dropout=0.1):
        super(feed_forward_network).__init__()
        self.linear1= nn.Linear(d_model, ff_hidden_dim)
        self.linear2= nn.Linear(ff_hidden_dim, d_model)
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(dropout)
    
    def forward(self, x):
        x= self.linear1(x)
        x= self.relu(x)
        x= self.dropout(x)
        x= self.linear2(x)
        return x

class Encoder_Layer(nn.Module):
    def __init__(self, d_model, ff_hidden_dim, n_heads, dropout):
        super(Encoder_Layer, self).__init__()
        self.mha= multi_head_attention(d_model, n_heads)
        self.normalization1= layer_normalization([d_model]) # passing as a list to ensure not being integer.
        self.dropout1= nn.Dropout(dropout)
        #self.ffn= nn.Sequential(
        self.normalization2= layer_normalization([d_model])
        self.dropout2= nn.Dropout(dropout)

    def forward(self, x):
        x_copy= x
        x= self.mha(x,mask=None)
        x= self.dropout1(x)
        x= self.normalization1(x+x_copy) # residual connection
        x_copy= x
        x= self.ffn(x)
        x= self.dropout2(x)
        x= self.normalization2(x+x_copy) # residual connection
        return x


# main encoder layer, which consists of multi-head attention and feed forward network, 
# with layer normalization and residual connections.
class Encoder(nn.Module):
    def __init__(self, d_model,ff_hidden_dim,  n_heads, dropout,n_layers):
        super().__init__()
        self.layers= nn.Sequential(*[Encoder_Layer(d_model,ff_hidden_dim, n_heads, dropout) for i in range(n_layers) ])
        # we specifically use nn.Sequential here bcs we can stack the encoder layers on top of each other. 
        # we use * to pass encoder layers as a list of arguments to nn.Sequential.

    def forward(self, x):
        x=self.layers(x)
        return x