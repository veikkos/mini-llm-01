# 06 — Mini LLM with CUDA

Same transformer architecture as 05, but running entirely on the GPU via custom CUDA kernels.
Includes BPE tokenizer, AdamW optimizer, and TinyStories training pipeline.

## Requirements

- NVIDIA GPU (tested on RTX 4080)
- CUDA Toolkit installed (`nvcc` in PATH)
- Visual Studio Developer Command Prompt (Windows)
- Node.js

## Quick Start

```bash
# Build (from Developer Command Prompt)
build.cmd

# Download TinyStories dataset (~250MB subset)
node download-tinystories.js

# Train BPE vocabulary (use --maxchars to limit corpus size)
node train-vocab.js --input ../data/tinystories.txt --maxchars 10000000

# Pre-encode text to binary tokens (one-time, ~5 min for 250MB)
node encode.js --input ../data/tinystories.txt

# Train (loads pre-encoded tokens instantly)
node train.js --tokens tokens.bin --steps 5000

# Train longer for better quality
node train.js --tokens tokens.bin --steps 50000

# Generate text
node generate.js
node generate.js --seed "The little dog" --temp 0.8
```

## Training Pipeline

1. **download-tinystories.js** — streams first 250MB of TinyStories from HuggingFace
2. **train-vocab.js** — trains BPE vocabulary (default 4096 tokens)
3. **encode.js** — pre-encodes text into binary token file with progress/ETA
4. **train.js** — trains the model using AdamW with warmup + cosine LR schedule
5. **generate.js** — generates text from trained weights

## Architecture

Default config: embedDim=256, 6 layers, 8 heads, contextLen=128 → ~6.8M parameters

Everything runs on GPU:
- All tensor data lives in CUDA memory
- Forward pass: gather → add → attention → ffn → logits (all GPU kernels)
- Backward pass: explicit gradient computation (no autograd graph overhead)
- AdamW optimizer with per-parameter momentum and variance buffers on GPU
- Only loss values and generated tokens come back to CPU

## CUDA Kernels

Custom kernels in `native/kernels.cu`:
- `matmul_kernel` / `matmul_add_kernel` — matrix multiply (forward + backward)
- `attn_scores_kernel` — Q @ K^T with causal mask
- `softmax_kernel` / `softmax_backward_kernel` — per-row softmax
- `log_softmax_kernel` / `log_softmax_backward_kernel` — output layer
- `tanh_kernel` / `tanh_backward_kernel` — activation
- `gather_kernel` / `gather_backward_kernel` — embedding lookup
- `concat_heads_kernel` / `split_heads_kernel` — multi-head attention concat/split
- `nll_loss_kernel` / `nll_loss_backward_kernel` — loss computation
- `adamw_kernel` — AdamW parameter updates with bias correction
- `transpose_kernel`, `copy_kernel`, `add_inplace_kernel` — utilities

## Changing GPU architecture

Edit `build.cmd` and change `-arch=sm_89` to match your GPU:
- RTX 4080/4090: `sm_89`
- RTX 3080/3090: `sm_86`
- RTX 2080: `sm_75`
- GTX 1080: `sm_61`
