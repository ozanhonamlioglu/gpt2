import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

n_vocab    = 50257 # GPT2 vocab size
embed_dim  =   256 # embedding dimension
seq_len    =   256 # max sequence length
n_heads    =   8 # attention heads
n_blocks   =   8 # transformer blocks
batch_size =   32
dropout    =   0.1 # standard GPT-2 10% dropout rate

device = "cuda" if torch.cuda.is_available() else "mps"