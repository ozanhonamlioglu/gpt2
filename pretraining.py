from databuilder import get_tinystories
from torch.utils.data import Dataset, DataLoader
import meta
import train
import model

storyTokens = get_tinystories()

class TokenDataset(Dataset):
  def __init__(self, tokens, seq_len):

    # initialize
    self.inputs  = []
    self.targets = []

    stride = meta.seq_len
    
    for i in range(0,len(tokens)-seq_len, stride):
      self.inputs.append( tokens[i   : i+seq_len])
      self.targets.append(tokens[i+1 : i+seq_len+1])

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return self.inputs[idx], self.targets[idx]

token_dataset= TokenDataset(storyTokens, meta.seq_len)
loader = DataLoader(token_dataset, batch_size=meta.batch_size, shuffle=True)
gpt2 = model.GPT2().to(meta.device)

train.run("gpt2_tinystories.pt", gpt2, loader, lr=3e-4)