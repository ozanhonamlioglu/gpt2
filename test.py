import torch
import meta
import model

gpt2 = model.GPT2().to(meta.device)
gpt2.load_state_dict(torch.load("gpt2_qa.pt", map_location=meta.device, weights_only=True))
gpt2.eval()

print("Ask a question (type 'exit' or press Ctrl+C to quit).\n")

while True:
  try:
    question = input("You: ").strip()
  except (EOFError, KeyboardInterrupt):
    print("\nBye!")
    break

  if not question:
    continue
  if question.lower() in ("exit", "quit"):
    print("Bye!")
    break

  prompt = f"Question: {question}\nAnswer:"
  print("GPT2:", end="", flush=True)
  try:
    _, ppl = gpt2.generate(prompt, generate_hook=lambda t: print(t, end="", flush=True), temperature=0.5)
  except KeyboardInterrupt:
    print("\n[interrupted]\n")
    continue

  print(f"\n[Perplexity: {ppl:.2f}]\n")