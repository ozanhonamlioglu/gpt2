from model import GPT2
from databuilder import storyTokens, tokenizer
import model

import torch
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "mps"

class TokenDataset(Dataset):
  def __init__(self, tokens, seq_len):

    # initialize
    self.inputs  = []
    self.targets = []

    stride = model.seq_len
    
    for i in range(0,len(tokens)-seq_len, stride):
      self.inputs.append( tokens[i   : i+seq_len])
      self.targets.append(tokens[i+1 : i+seq_len+1])

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return self.inputs[idx], self.targets[idx]


token_dataset= TokenDataset(storyTokens.to(device), model.seq_len)
loader = DataLoader(token_dataset, batch_size=model.batch_size, shuffle=True)

X, y = next(iter(loader))
gpt2 = model.GPT2().to(device)

epoch = 3

lossfun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(gpt2.parameters(), lr=3e-4, weight_decay=0.01)

for ep in range(epoch):
  for step, (X, y) in enumerate(loader):
    pred = gpt2(X)
    loss = lossfun(pred.view(-1, model.n_vocab), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
      print(f"Epoch {ep+1} | Step {step}/{len(loader)} | Loss: {loss.item():.4f}")

torch.save(gpt2.state_dict(), "gpt2_tinystories.pt")