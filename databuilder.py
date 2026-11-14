import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import meta

def get_tinystories():
    if os.path.exists("tinystories.pt"):
        storyTokens = torch.load("tinystories.pt", weights_only=True)
    else:
        dataset = load_dataset("roneneldan/TinyStories", split="train[:200000]")
        all_tokens = []
        for item in tqdm(dataset, desc="Tokenizing TinyStories"):
            all_tokens.extend(meta.tokenizer.encode(item["text"]))
            all_tokens.append(meta.tokenizer.eos_token_id)
        storyTokens = torch.tensor(all_tokens, dtype=torch.long)
        torch.save(storyTokens, "tinystories.pt")
    
    return storyTokens

def get_alpacaQA():
    if os.path.exists("alpaca.pt"):
        qa_data = torch.load("alpaca.pt", weights_only=True)
    else:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        inputs, targets = [], []

        for item in tqdm(dataset, desc="Tokenizing Alpaca"):
            if item["input"]:
                continue

            # 1. Format Question and Answer
            q_text = f"Question: {item['instruction']}\nAnswer:"
            a_text = f" {item['output']}"

            q_ids = meta.tokenizer.encode(q_text)
            a_ids = meta.tokenizer.encode(a_text) + [meta.tokenizer.eos_token_id]

            full = q_ids + a_ids
            if len(full) > meta.seq_len + 1:
                continue  # skip examples longer than seq_len

            # 2. Shift by 1 to create X and y
            x = full[:-1]
            y = full[1:]

            # 3. Mask the Question part in y with -100 (only grade the Answer!)
            q_mask_len = len(q_ids) - 1
            y[:q_mask_len] = [-100] * q_mask_len

            # 4. Pad X with <eos> and y with -100 up to seq_len
            pad_len = meta.seq_len - len(x)
            x = x + [meta.tokenizer.eos_token_id] * pad_len
            y = y + [-100] * pad_len

            inputs.append(x)
            targets.append(y)

        qa_data = {
        "inputs": torch.tensor(inputs, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        }
        torch.save(qa_data, "alpaca.pt")
    
    return qa_data