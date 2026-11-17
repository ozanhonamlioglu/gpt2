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
    B, T, C = x.shape
    qkv = self.qkv_layer(x)
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    head_dim = C // meta.n_heads

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

    # GPT-2 style init (std 0.02); PyTorch's default N(0,1) embedding makes tied logits huge at start
    self.apply(self._init_weights)
    for name, p in self.named_parameters():
      if name.endswith("mlp_contract.weight") or name.endswith("w0.weight"):
        # scale residual projections down so the residual stream doesn't grow with depth
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * meta.n_blocks) ** 0.5)

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
      torch.nn.init.zeros_(module.bias)
  
  def forward(self, tokx):
    token_embed = self.embedding(tokx)
    posit_embed = self.positions(torch.arange(tokx.shape[-1], device=tokx.device))
    x = self.dropout(token_embed + posit_embed)
    x = self.blocks(x)
    x = self.layernormC(x)
    return self.finalLinear(x)
  
  @torch.no_grad()
  def generate(
      self,
      prompt,
      generate_hook=None, # hook for printing while generating
      max_new_tokens=100,
      temperature=0.7,
      top_k=40,   # only sample from the 40 most likely tokens
      top_p=0.9,  # ...and only from the smallest set covering 90% of the probability
      repetition_penalty=1.2, # make tokens already used in the answer less likely (1.0 = off)
      no_repeat_ngram_size=3  # never repeat the same 3-token phrase in the answer (0 = off, min 2)
    ):
    tokens = meta.tokenizer.encode(prompt, return_tensors="pt").to(meta.device)
    prompt_len = tokens.shape[1]

    nlls = [] # store negative log-likelihood of each generated token

    for _ in range(max_new_tokens):
      idx_cond = tokens[:, -meta.seq_len:]
      logits = self(idx_cond)
      last_logits = logits[:, -1, :].clone()

      # only look at the answer, so the model can still reuse words from the question
      answer = tokens[0, prompt_len:].tolist()

      # repetition penalty (CTRL-style): shrink the logit of every token already in the answer
      if repetition_penalty != 1.0 and answer:
        used = torch.tensor(sorted(set(answer)), device=tokens.device)
        scores = last_logits[0, used]
        last_logits[0, used] = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)

      # no-repeat n-gram: if the last n-1 tokens appeared before, ban the token that followed them
      n = no_repeat_ngram_size
      if n > 1 and len(answer) >= n:
        prefix = answer[-(n - 1):]
        banned = {answer[i + n - 1] for i in range(len(answer) - n + 1) if answer[i : i + n - 1] == prefix}
        last_logits[0, list(banned)] = float("-inf")

      last_logits = last_logits / temperature
      if top_k is not None:
        kth = torch.topk(last_logits, top_k, dim=-1).values[:, -1, None]
        last_logits = last_logits.masked_fill(last_logits < kth, float("-inf"))
      if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(last_logits, descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum - F.softmax(sorted_logits, dim=-1) > top_p  # always keep the top token
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        last_logits = torch.full_like(last_logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
      probs = F.softmax(last_logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
      
      # Grab the probability of the token we just picked (e.g. probs[0, 2402])
      picked_prob = probs[0, next_token.item()]
      nlls.append(-torch.log(picked_prob))
      
      # Stop if the model outputs <|endoftext|> (50256)
      if next_token.item() == meta.tokenizer.eos_token_id:
        break

      if generate_hook is not None:
        generate_hook(meta.tokenizer.decode(next_token[0]))
      
      tokens = torch.cat((tokens, next_token), dim=1)
    
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return tokens, ppl