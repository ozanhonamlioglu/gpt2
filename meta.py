import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

n_vocab    = 50257 # GPT2 vocab size
embed_dim  =   256 # embedding dimension
seq_len    =   256 # max sequence length
n_heads    =   8 # attention heads
n_blocks   =   8 # transformer blocks
batch_size =   8

###################################
# ENABLE BELOW TO MORE PRECISION #
##################################
# n_vocab    = 50257 # GPT2 vocab size
# embed_dim  =   512 # 2x wider (gives standard 64-dim attention heads: 512 / 8 = 64)
# seq_len    =   256 # fits full stories & Q&A while keeping attention fast
# n_heads    =     8 # 8 heads x 64 head_dim
# n_blocks   =     8 # 8 transformer blocks
# batch_size =    32 # 4x bigger batch = much more stable gradients & 4x fewer steps/epoch

device = "cuda" if torch.cuda.is_available() else "mps"