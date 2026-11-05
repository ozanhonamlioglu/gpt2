# pytorch stuff
import torch
import torch.nn as nn
from torch.nn import functional as F

#n load GPT2 tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# data hyperparameters
seq_len = 8 # aka context window
n_vocab = tokenizer.vocab_size #n

# model hyperparameters
embed_dim = 2**8 # 256
n_layers = 12 # number of transformer blocks
mlp_scalar = 4 # times

# training hyperparameters
batch_size = 5

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.mlp_expand = nn.Linear(embed_dim, embed_dim * mlp_scalar)
    self.mlp_contract = nn.Linear(embed_dim * mlp_scalar, embed_dim)
  
  def forward(self, x):
    return self.mlp_contract(F.gelu(self.mlp_expand(x)))

class OneAttentionHead(nn.Module):
  def __init__(self):
    super().__init__()
    self.qkv_layer = nn.Linear(embed_dim, embed_dim * 3, bias=False)

  def forward(self, x):  
    qkv = self.qkv_layer(x)
    q,k,v = torch.chunk(qkv, 3, dim=-1)
    return F.scaled_dot_product_attention(q,k,v,is_causal=True)

class TransformerBlock(nn.Module):
  def __init__(self):
    super().__init__()

    # layer normalization
    self.layernormA = nn.LayerNorm(embed_dim) # before attention
    self.layernormB = nn.LayerNorm(embed_dim) # before MLP
    
    # attention header
    self.attn = OneAttentionHead()
    
    # MLP (feed-forward): expand 4x, nonlinearity, contract back
    self.mlp = MLP()

  def forward(self, x):
    # --- attention sublayer ---
    x_norm = self.layernormA(x)
    attn_output = self.attn(x_norm)
    x += attn_output
    
    # --- MLP sublayer ---
    x_norm = self.layernormB(x)
    mlp_output = self.mlp(x_norm)
    x += mlp_output # residual

    return x
  
class GPT2(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.embedding = nn.Embedding(n_vocab, embed_dim)
    self.positions = nn.Embedding(seq_len, embed_dim)
    
    self.layernormC = nn.LayerNorm(embed_dim) # before final
    
    # transformer blocks, applied one after another
    self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layers)])
    
    # final layer
    self.finalLinear = nn.Linear(embed_dim, n_vocab, bias=False)
    
    # tied output
    self.finalLinear.weight = self.embedding.weight
  
  def forward(self, tokx):
    token_embed = self.embedding(tokx)
    posit_embed = self.positions(torch.arange(tokx.shape[-1]))
    
    x = token_embed + posit_embed
    
    x = self.blocks(x)
    
    x = self.layernormC(x)

    return self.finalLinear(x)