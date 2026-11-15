# GPT-2 From Scratch in PyTorch

A modular, end-to-end implementation of the **GPT-2** transformer architecture built from scratch in PyTorch—featuring a complete two-stage LLM training pipeline: **Stage 1 Pre-Training** (learning language & grammar) and **Stage 2 Post-Training / Supervised Fine-Tuning** (learning Question-Answering).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r install.txt
```

### 2. Configure Hyperparameters (`meta.py`)
Adjust model dimensions (`embed_dim`, `seq_len`, `n_heads`, `n_blocks`, `batch_size`) in `meta.py` to match your hardware (e.g., compact preset for local Apple Silicon `mps` or full 124M preset for cloud GPUs).

### 3. Stage 1: Pre-Training (Base Model)
Train the base GPT-2 model from scratch on raw narrative text (`TinyStories`) so it learns English grammar, vocabulary, and next-token prediction:
```bash
python pretraining.py
```
- Tokenizes and caches the corpus into `tinystories.pt` with `<|endoftext|>` document boundaries.
- Trains from random weights (`lr = 3e-4`) and exports the base model checkpoint to **`gpt2_tinystories.pt`**.

### 4. Stage 2: Post-Training / Supervised Fine-Tuning (Q&A Model)
Transform the pre-trained base model into a Question-Answering assistant using instruction data (`Alpaca`):
```bash
python supertuning.py
```
- Formats Q&A pairs (`Question: ...\nAnswer: ...<|endoftext|>`) and caches them into `alpaca.pt`.
- Applies **`-100` target loss masking** to the question and padding tokens so gradients exclusively train the model on generating answers.
- Loads **`gpt2_tinystories.pt`**, fine-tunes with a gentle learning rate (`lr = 5e-5`), and exports the post-trained checkpoint to **`gpt2_qa.pt`**.

### 5. Run Inference (`test.py`)
Generate answers and evaluate model confidence (perplexity):
```bash
python test.py
```

---

## Project Structure

```text
├── meta.py          # Hyperparameters, tokenizer, and device configuration
├── model.py         # GPT-2 architecture & built-in generate() method
├── databuilder.py   # Stage 1 (TinyStories) & Stage 2 (Alpaca Q&A) data builders
├── train.py         # Shared bfloat16 mixed-precision training engine
├── pretraining.py   # Stage 1: Base language model pre-training
├── supertuning.py   # Stage 2: Post-training / Supervised Fine-Tuning (SFT)
├── test.py          # Q&A inference and perplexity evaluation
└── install.txt      # Core Python dependencies
```

## Core Highlights

- **Two-Stage LLM Pipeline (`pretraining.py` $\rightarrow$ `supertuning.py`)**
  - Mirrors modern production LLM workflows by separating **Pre-Training** (continuous document packing on `TinyStories`) from **Post-Training / SFT** (prompt-masked instruction tuning on `Alpaca`).
- **Custom GPT-2 Architecture (`model.py`)**
  - Causal **Multi-Head Self-Attention** (`scaled_dot_product_attention`), Pre-LayerNorm **Transformer Blocks**, GELU `MLP` layers, dropout regularization, and **weight tying** between token embeddings and the output projection head.
  - Built-in `@torch.no_grad()` autoregressive `generate()` method with temperature control, live streaming hooks, and automatic **perplexity** scoring.
- **Fast Mixed-Precision Training (`train.py` & `databuilder.py`)**
  - Uses `bfloat16` mixed precision (`torch.autocast`) with `AdamW` and `CrossEntropyLoss`, paired with automatic local `.pt` dataset caching for instant startup on repeated runs.
