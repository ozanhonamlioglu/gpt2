import torch
from torch.utils.data import TensorDataset, DataLoader
import model
from databuilder import get_alpacaQA
import meta
import train

qa_data = get_alpacaQA()

# 1. Wrap inputs & targets into a Dataset and DataLoader (1 line each!)
qa_dataset = TensorDataset(qa_data["inputs"], qa_data["targets"])
loader = DataLoader(qa_dataset, batch_size=meta.batch_size, shuffle=True)

# 2. Load your Pre-Trained Stage 1 Model!
gpt2 = model.GPT2().to(meta.device)
gpt2.load_state_dict(torch.load("gpt2_tinystories.pt", map_location=meta.device, weights_only=True))

train.run("gpt2_qa.pt", gpt2, loader, lr=5e-5)