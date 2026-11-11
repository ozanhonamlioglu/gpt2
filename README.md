# GPT-2 From Scratch in PyTorch

A lightweight, modular implementation of the **GPT-2** transformer architecture built from scratch in PyTorch—covering dataset tokenization, pre-training, and real-time streaming inference.

## Features

- **Custom GPT-2 Architecture (`model.py`)**
  - Causal **Multi-Head Self-Attention** (plus an optional single-head attention module) powered by PyTorch's `scaled_dot_product_attention`.
  - Pre-LayerNorm **Transformer Blocks** with residual connections and GELU feed-forward (`MLP`) networks.
  - Learned token & positional embeddings with **weight tying** between the token embedding layer and final language modeling head.
- **Centralized Configuration (`meta.py`)**
  - Single source of truth for model dimensions (`embed_dim`, `seq_len`, `n_heads`, `n_blocks`, `batch_size`), tokenizer setup, and automatic hardware acceleration (`cuda` / `mps`).
- **Automated Data Pipeline (`databuilder.py`)**
  - Streams/loads text corpora via Hugging Face `datasets` (`TinyStories` by default).
  - Appends `<|endoftext|>` boundary tokens between documents and caches the packed 1D token tensor locally (`.pt`) for instant startup on subsequent runs.
- **End-to-End Training (`train.py`)**
  - Non-overlapping context window chunking via a custom PyTorch `Dataset` & `DataLoader`.
  - Next-token prediction training loop using `CrossEntropyLoss` and `AdamW`, exporting trained weights as a portable `.pt` checkpoint.
- **Streaming Autoregressive Inference (`infer.py`)**
  - Loads trained checkpoints across devices (`CUDA` or Apple Silicon `MPS`) and streams generated tokens live to the terminal with configurable temperature sampling and automatic `<|endoftext|>` stopping.

## Project Structure

```text
├── meta.py          # Hyperparameters, tokenizer, and device configuration
├── model.py         # GPT-2 architecture (Attention, MLP, TransformerBlock, GPT2)
├── databuilder.py   # Dataset downloading, tokenization, and local .pt caching
├── train.py         # Dataset chunking, DataLoader, and training loop
├── infer.py         # Live token-by-token text generation
└── install.txt      # Core Python dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r install.txt
```

### 2. Configure Hyperparameters
Adjust model scale and batch size in `meta.py` to match your hardware (e.g., compact preset for local Apple Silicon `mps` or scaled-up preset for cloud GPUs).

### 3. Train the Model
```bash
python train.py
```
On the first run, `databuilder.py` automatically downloads and tokenizes the dataset into `tinystories.pt`, then trains the model and saves the weights to `gpt2_tinystories.pt`.

### 4. Generate Text
```bash
python infer.py
```
Loads `gpt2_tinystories.pt` and streams generated completions directly to the terminal.

## Roadmap & Upcoming Features

- **Post-Training / Supervised Fine-Tuning (SFT)**
  - Second-stage training pipeline on instruction and Question-Answering datasets (e.g., **Alpaca**).
  - Structured prompt templates (`Question: ... Answer: ...<|endoftext|>`) and target loss masking so the model learns to directly answer user questions rather than simply continuing stories.
- **Interactive Q&A Mode**
  - Conversational CLI loop in `infer.py` formatted for direct question-and-answer interactions.

