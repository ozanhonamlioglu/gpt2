import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Parameters are adjusted for colab A100 GPU
# n_vocab    =    50257 # GPT2 vocab size
# embed_dim  =    768
# seq_len    =    512
# n_heads    =    12
# n_blocks   =    12
# batch_size =    32
# grad_accum =    1
# dropout    =    0.1

# GTX 1050 Ti (4 GB, fp32): measured ~4.7k tok/s, ~2.3 GB peak at batch 8
# ~51M params (25.7M embedding + 25.2M transformer); ~8 h for one pass over the pretraining mix
n_vocab    =    50257 # GPT2 vocab size
embed_dim  =    512 # embedding dimension (head_dim = 512 / 8 = 64, same as real GPT-2)
seq_len    =    256 # max sequence length (alpaca.pt is built for this; delete it if you change this)
n_heads    =    8 # attention heads
n_blocks   =    8 # transformer blocks
batch_size =    8 # sequences per forward pass (what fits in memory)
grad_accum =    4 # optimizer step every 4 batches -> effective batch 32 x 256 = 8k tokens
dropout    =    0.1 # used for SFT; pretraining.py sets it to 0.0

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
