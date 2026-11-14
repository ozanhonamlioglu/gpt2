import torch
import torch.nn.functional as F
import meta
import model

gpt2 = model.GPT2().to(meta.device)
gpt2.load_state_dict(torch.load("gpt2_tinystories.pt", map_location=meta.device, weights_only=True))
gpt2.eval()

prompt = "Once upon a time there was a little"
gen, ppl = gpt2.generate(prompt, max_new_tokens=300)

print(meta.tokenizer.decode(gen[0]))
print()
print(f"\n[Perplexity: {ppl:.2f}]")