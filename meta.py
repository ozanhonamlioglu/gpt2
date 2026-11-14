import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Parameters are adjusted for colab A100 GPU
n_vocab    =    50257 # GPT2 vocab size
embed_dim  =    768
seq_len    =    512
n_heads    =    12
n_blocks   =    12
batch_size =    32
dropout    =    0.1

device = "cuda" if torch.cuda.is_available() else "mps"