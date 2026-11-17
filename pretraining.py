import torch
from databuilder import get_pretrain_mix
from torch.utils.data import Dataset, DataLoader, random_split
import meta

# No dropout for pretraining: one pass over ~140M tokens, the model can't overfit.
# (set before importing model/train so every layer picks it up)
meta.dropout = 0.0

import train
import model

tokens = get_pretrain_mix()
print(f"Pretraining tokens: {len(tokens):,}")

class TokenDataset(Dataset):
  def __init__(self, tokens, seq_len):
    self.tokens = tokens
    self.seq_len = seq_len
    self.n = (len(tokens) - 1) // seq_len  # non-overlapping windows

  def __len__(self):
    return self.n

  def __getitem__(self, idx):
    i = idx * self.seq_len
    return self.tokens[i : i + self.seq_len], self.tokens[i + 1 : i + self.seq_len + 1]

token_dataset = TokenDataset(tokens, meta.seq_len)
val_size = 2000  # ~0.5M tokens held out to track generalization
train_set, val_set = random_split(token_dataset, [len(token_dataset) - val_size, val_size],
                                  generator=torch.Generator().manual_seed(0))

loader     = DataLoader(train_set, batch_size=meta.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=meta.batch_size)

gpt2 = model.GPT2().to(meta.device)
train.run("gpt2_pretrain.pt", gpt2, loader, lr=6e-4, epoch=1, val_loader=val_loader, warmup_steps=500)
