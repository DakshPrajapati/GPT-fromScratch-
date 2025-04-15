import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention 
    parameters:
        - head_size
        - n_embd
        - block_size
        - dropout
    """

    def __init__(self, head_size,n_embd,block_size,dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out



class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel 
    parameters:
        - num_heads
        - head_size
        - n_embd
        - dropout
        - block_size
    """
    

    def __init__(self, num_heads, head_size,n_embd,dropout,block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,n_embd,block_size,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity 
    parameters:
        - n_embd
        - dropout
    """

    def __init__(self, n_embd,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation 
    Parameters:
        - n_embed
        - n_head
        - n_embd
        - dropout
        - block_size
    """

    def __init__(self,
                 n_embd,
                 n_head,
                 dropout,
                 block_size):
        
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,
                                     head_size,
                                     n_embd,
                                     dropout, 
                                     block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
