import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import meta

if os.path.exists("tinystories.pt"):
    storyTokens = torch.load("tinystories.pt", weights_only=True)
else:
    dataset = load_dataset("roneneldan/TinyStories", split="train[:100000]")
    all_tokens = []
    for item in tqdm(dataset, desc="Tokenizing TinyStories"):
        all_tokens.extend(meta.tokenizer.encode(item["text"]))
        all_tokens.append(meta.tokenizer.eos_token_id)
    storyTokens = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(storyTokens, "tinystories.pt")