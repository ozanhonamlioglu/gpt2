import torch
import torch.nn.functional as F
import meta
import model

gpt2 = model.GPT2().to(meta.device)
gpt2.load_state_dict(torch.load("gpt2_qa.pt", map_location=meta.device, weights_only=True))
gpt2.eval()

question = "Count from 1 to 10"
prompt = f"Question: {question}\nAnswer:"
gen, ppl = gpt2.generate(prompt)

print(meta.tokenizer.decode(gen[0]))
print()
print(f"\n[Perplexity: {ppl:.2f}]")