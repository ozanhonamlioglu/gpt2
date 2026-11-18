import math
import time
import torch
import meta

def configure_optimizer(gpt2, lr, weight_decay):
  # decay matrices (linear weights, embeddings); don't decay biases and LayerNorm
  decay   = [p for p in gpt2.parameters() if p.requires_grad and p.dim() >= 2]
  nodecay = [p for p in gpt2.parameters() if p.requires_grad and p.dim() < 2]
  return torch.optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": nodecay, "weight_decay": 0.0}],
    lr=lr, betas=(0.9, 0.95),
  )

def lr_at(step, total_steps, max_lr, warmup_steps, min_lr_ratio=0.1):
  # linear warmup, then cosine decay down to min_lr_ratio * max_lr
  if step < warmup_steps:
    return max_lr * (step + 1) / warmup_steps
  progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
  return max_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

@torch.no_grad()
def evaluate(gpt2, loader):
  gpt2.eval()
  total, count = 0.0, 0
  for X, y in loader:
    X, y = X.to(meta.device).long(), y.to(meta.device).long()
    pred = gpt2(X)
    total += torch.nn.functional.cross_entropy(
      pred.view(-1, meta.n_vocab), y.view(-1), ignore_index=-100, reduction="sum").item()
    count += (y != -100).sum().item()
  gpt2.train()
  return total / max(1, count)

def run(outputfile, gpt2, loader, lr, epoch=2, val_loader=None,
        grad_accum=meta.grad_accum, warmup_steps=200, weight_decay=0.1, log_every=50):

  # print model information
  total_params = sum(p.numel() for p in gpt2.parameters())
  print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
  print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")

  lossfun = torch.nn.CrossEntropyLoss(ignore_index=-100)
  optimizer = configure_optimizer(gpt2, lr, weight_decay)

  steps_per_epoch = len(loader) // grad_accum
  total_steps = steps_per_epoch * epoch
  print(f"Optimizer steps: {total_steps:,} ({steps_per_epoch:,}/epoch, "
        f"effective batch {meta.batch_size * grad_accum} sequences)")

  gpt2.train()
  step = 0
  best_val = float("inf")
  for ep in range(epoch):
    epoch_loss, epoch_batches = 0.0, 0
    window_loss, window_batches = 0.0, 0
    t0 = time.time()

    for micro, (X, y) in enumerate(loader):
      X, y = X.to(meta.device).long(), y.to(meta.device).long()
      pred = gpt2(X)
      loss = lossfun(pred.view(-1, meta.n_vocab), y.view(-1))
      (loss / grad_accum).backward()

      epoch_loss += loss.item(); epoch_batches += 1
      window_loss += loss.item(); window_batches += 1

      if (micro + 1) % grad_accum != 0:
        continue

      for g in optimizer.param_groups:
        g["lr"] = lr_at(step, total_steps, lr, warmup_steps)
      torch.nn.utils.clip_grad_norm_(gpt2.parameters(), 1.0)
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      step += 1

      if step % log_every == 0:
        elapsed = time.time() - t0
        tok_s = window_batches * meta.batch_size * meta.seq_len / elapsed
        remaining = (total_steps - step) * grad_accum * meta.batch_size * meta.seq_len / tok_s
        print(f"Epoch {ep+1} | Step {step}/{total_steps} | Loss (avg) {window_loss / window_batches:.4f} "
              f"| lr {optimizer.param_groups[0]['lr']:.2e} | {tok_s:,.0f} tok/s | ETA {remaining / 3600:.1f}h")
        window_loss, window_batches = 0.0, 0
        t0 = time.time()

    print(f"=== Epoch {ep+1} done | train loss {epoch_loss / max(1, epoch_batches):.4f}", end="")
    if val_loader is not None:
      val = evaluate(gpt2, val_loader)
      print(f" | val loss {val:.4f}", end="")
      torch.save(gpt2.state_dict(), outputfile.replace(".pt", f"_ep{ep+1}.pt"))
      if val < best_val:
        best_val = val
        torch.save(gpt2.state_dict(), outputfile)
        print(" | best so far, saved", end="")
    print(" ===")

  if val_loader is None:
    torch.save(gpt2.state_dict(), outputfile)
  else:
    print(f"Best val loss {best_val:.4f} -> {outputfile}")
