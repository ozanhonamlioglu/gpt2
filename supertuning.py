import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import model
from databuilder import get_sft_mix
import meta
import train

qa_data = get_sft_mix()

# 1. Wrap inputs & targets (Alpaca + Dolly) into a Dataset, hold out 5% to pick the best epoch
qa_dataset = TensorDataset(qa_data["inputs"], qa_data["targets"])
val_size = len(qa_dataset) // 20
train_set, val_set = random_split(qa_dataset, [len(qa_dataset) - val_size, val_size],
                                  generator=torch.Generator().manual_seed(0))
loader     = DataLoader(train_set, batch_size=meta.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=meta.batch_size)

# 2. Load your Pre-Trained Stage 1 Model!
gpt2 = model.GPT2().to(meta.device)
gpt2.load_state_dict(torch.load("gpt2_pretrain.pt", map_location=meta.device, weights_only=True))

# 3. Fine-tune; saves gpt2_qa_ep{N}.pt every epoch and the best (lowest val loss) as gpt2_qa.pt
train.run("gpt2_qa.pt", gpt2, loader, lr=3e-4, epoch=5, val_loader=val_loader, warmup_steps=100)
