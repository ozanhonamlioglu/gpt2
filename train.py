import model
import meta
import torch

def run(outputfile, gpt2, loader, lr, epoch=2):

  # print model information
  total_params = sum(p.numel() for p in gpt2.parameters())
  print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
  print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")

  lossfun = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(gpt2.parameters(), lr=lr, weight_decay=0.01)

  for ep in range(epoch):
    for step, (X, y) in enumerate(loader):
      X, y = X.to(meta.device), y.to(meta.device)
      with torch.autocast(device_type=meta.device, dtype=torch.bfloat16):
        pred = gpt2(X)
        loss = lossfun(pred.view(-1, meta.n_vocab), y.view(-1))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      if step % 50 == 0:
        print(f"Epoch {ep+1} | Step {step}/{len(loader)} | Loss: {loss.item():.4f}")

  torch.save(gpt2.state_dict(), outputfile)