import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters for GPT2-124M
n_vocab    = 50257 # GPT2 vocab size
embed_dim  =   256 # embedding dimension
seq_len    =   256 # max sequence length
n_heads    =   8 # attention heads
n_blocks   =   8 # transformer blocks
batch_size =   8

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.mlp_expand = nn.Linear(embed_dim, embed_dim * 4)
    self.mlp_contract = nn.Linear(embed_dim * 4, embed_dim)
  
  def forward(self, x):
    return self.mlp_contract(F.gelu(self.mlp_expand(x)))

class OneAttentionHead(nn.Module):
  def __init__(self):
    super().__init__()
    self.qkv_layer = nn.Linear(embed_dim, embed_dim * 3, bias=False)
    self.w0 = nn.Linear(embed_dim, embed_dim, bias=False)

  def forward(self, x):  
    qkv = self.qkv_layer(x)
    q,k,v = torch.chunk(qkv, 3, dim=-1)
    out = F.scaled_dot_product_attention(q,k,v,is_causal=True)
    return self.w0(out)

class MultiHeadAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.qkv_layer = nn.Linear(embed_dim, embed_dim * 3, bias=False)
    self.w0 = nn.Linear(embed_dim, embed_dim, bias=False)
    
  def forward(self, x):
    B, T, C = x.shape                       # (batch, seq, embed_dim)
    qkv = self.qkv_layer(x)                 # (B, T, 3C)
    q, k, v = torch.chunk(qkv, 3, dim=-1)   # each (B, T, C)
    head_dim = C // n_heads               # 256 / 8 = 32

    # (B, T, C) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, heads, T, head_dim)
    
    # merge heads back: (B, heads, T, head_dim) -> (B, T, C)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.w0(out)

class TransformerBlock(nn.Module):
  def __init__(self, single_head_attn = False):
    super().__init__()

    # layer normalization
    self.layernormA = nn.LayerNorm(embed_dim) # before attention
    self.layernormB = nn.LayerNorm(embed_dim) # before MLP
    
    self.attn = OneAttentionHead() if single_head_attn else MultiHeadAttention()
    self.mlp = MLP()

  def forward(self, x):
    # --- attention sublayer ---
    x_norm = self.layernormA(x)
    attn_output = self.attn(x_norm)
    x = x + attn_output
    
    # --- MLP sublayer ---
    x_norm = self.layernormB(x)
    mlp_output = self.mlp(x_norm)
    x = x + mlp_output # residual

    return x
  
class GPT2(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.embedding = nn.Embedding(n_vocab, embed_dim)
    self.positions = nn.Embedding(seq_len, embed_dim)
    
    self.layernormC = nn.LayerNorm(embed_dim) # before final
    
    # transformer blocks, applied one after another
    self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_blocks)])
    
    # final layer
    self.finalLinear = nn.Linear(embed_dim, n_vocab, bias=False)
    
    # tied output
    self.finalLinear.weight = self.embedding.weight
  
  def forward(self, tokx):
    token_embed = self.embedding(tokx)
    posit_embed = self.positions(torch.arange(tokx.shape[-1], device=tokx.device))
    
    x = token_embed + posit_embed
    
    x = self.blocks(x)
    
    x = self.layernormC(x)

    return self.finalLinear(x)