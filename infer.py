import torch
import torch.nn.functional as F
import meta
import model

gpt2 = model.GPT2().to(meta.device)
gpt2.load_state_dict(torch.load("gpt2_tinystories.pt", map_location=meta.device, weights_only=True))
gpt2.eval()

@torch.no_grad()
def generate(
  prompt,
  generate_hook=None, # hook for printing while generating
  max_new_tokens=100,
  temperature=0.8,
  ):
  tokens = meta.tokenizer.encode(prompt, return_tensors="pt").to(meta.device)
  
  nlls = [] # store negative log-likelihood of each generated token
  
  for _ in range(max_new_tokens):
    idx_cond = tokens[:, -meta.seq_len:]
    logits = gpt2(idx_cond)
    last_logits = logits[:, -1, :] / temperature
    probs = F.softmax(last_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    # Grab the probability of the token we just picked (e.g. probs[0, 2402])
    picked_prob = probs[0, next_token.item()]
    nlls.append(-torch.log(picked_prob))
    
    # Stop if the model outputs <|endoftext|> (50256)
    if next_token.item() == meta.tokenizer.eos_token_id:
      break

    if generate_hook is not None:
      generate_hook(meta.tokenizer.decode(next_token[0]))
    
    tokens = torch.cat((tokens, next_token), dim=1)
  
  ppl = torch.exp(torch.stack(nlls).mean()).item()
  return tokens, ppl


prompt = "Once upon a time there was a little"
gen, ppl = generate(prompt, max_new_tokens=300)

print(meta.tokenizer.decode(gen[0]))
print()
print(f"\n[Perplexity: {ppl:.2f}]")