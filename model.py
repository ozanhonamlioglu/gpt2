import torch
import torch.nn as nn
from torch.nn import functional as F
import meta

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.mlp_expand = nn.Linear(meta.embed_dim, meta.embed_dim * 4)
    self.mlp_contract = nn.Linear(meta.embed_dim * 4, meta.embed_dim)
    self.dropout = nn.Dropout(meta.dropout)
  
  def forward(self, x):
    return self.dropout(self.mlp_contract(F.gelu(self.mlp_expand(x))))

class MultiHeadAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.qkv_layer = nn.Linear(meta.embed_dim, meta.embed_dim * 3, bias=False)
    self.w0 = nn.Linear(meta.embed_dim, meta.embed_dim, bias=False)
    self.dropout = nn.Dropout(meta.dropout)
    
  def forward(self, x):
    B, T, C = x.shape                       # (batch, seq, embed_dim)
    qkv = self.qkv_layer(x)                 # (B, T, 3C)
    q, k, v = torch.chunk(qkv, 3, dim=-1)   # each (B, T, C)
    head_dim = C // meta.n_heads               # 256 / 8 = 32

    # (B, T, C) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
    q = q.view(B, T, meta.n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, meta.n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, meta.n_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=meta.dropout if self.training else 0.0, is_causal=True)  # (B, heads, T, head_dim)
    
    # merge heads back: (B, heads, T, head_dim) -> (B, T, C)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.dropout(self.w0(out))

class TransformerBlock(nn.Module):
  def __init__(self):
    super().__init__()

    # layer normalization
    self.layernormA = nn.LayerNorm(meta.embed_dim) # before attention
    self.layernormB = nn.LayerNorm(meta.embed_dim) # before MLP
    
    self.attn = MultiHeadAttention()
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
    
    self.embedding = nn.Embedding(meta.n_vocab, meta.embed_dim)
    self.positions = nn.Embedding(meta.seq_len, meta.embed_dim)
    self.dropout = nn.Dropout(meta.dropout)
    
    self.layernormC = nn.LayerNorm(meta.embed_dim) # before final
    
    # transformer blocks, applied one after another
    self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(meta.n_blocks)])
    
    # final layer
    self.finalLinear = nn.Linear(meta.embed_dim, meta.n_vocab, bias=False)
    
    # tied output
    self.finalLinear.weight = self.embedding.weight
  
  def forward(self, tokx):
    token_embed = self.embedding(tokx)
    posit_embed = self.positions(torch.arange(tokx.shape[-1], device=tokx.device))
    
    x = self.dropout(token_embed + posit_embed)
    
    x = self.blocks(x)
    
    x = self.layernormC(x)

    return self.finalLinear(x)