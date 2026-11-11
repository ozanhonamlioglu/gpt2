import torch
import torch.nn.functional as F
import meta
import model

gpt2 = model.GPT2().to(meta.device)
gpt2.load_state_dict(torch.load("gpt2_tinystories.pt", map_location=meta.device, weights_only=True))
gpt2.eval()

@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=0.8):
  print(prompt, end="", flush=True) # print prompt first
  tokens = meta.tokenizer.encode(prompt, return_tensors="pt").to(meta.device)
  
  for _ in range(max_new_tokens):
    idx_cond = tokens[:, -meta.seq_len:]
    logits = gpt2(idx_cond)
    last_logits = logits[:, -1, :] / temperature
    probs = F.softmax(last_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    # Stop if the model outputs <|endoftext|> (50256)
    if next_token.item() == meta.tokenizer.eos_token_id:
      break

    word = meta.tokenizer.decode(next_token[0])
    print(word, end="", flush=True)
    
    tokens = torch.cat((tokens, next_token), dim=1)

generate("Why is the sky blue?")