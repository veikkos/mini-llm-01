# Mini LLM 01

*Unsuprisingly created with an AI agent as an educational study. Not production grade or anything but does something.*

A small transformer language model running entirely on the GPU via custom CUDA kernels.
Includes BPE tokenizer, AdamW optimizer, and TinyStories training pipeline.

## Requirements

- NVIDIA GPU (tested on RTX 4080)
- CUDA Toolkit installed (`nvcc` in PATH)
- Visual Studio Developer Command Prompt (Windows)
- Node.js

## Quick Start

```bash
# Build (from Developer Command Prompt, see GPU architecture section below)
build.cmd

# Download TinyStories dataset (~250MB subset)
node download-tinystories.js

# Train BPE vocabulary (use --maxchars to limit corpus size)
node train-vocab.js --input data/tinystories.txt

# Pre-encode text to binary tokens (one-time, ~5 min for 250MB)
node encode.js --input data/tinystories.txt

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

- Tensor data and optimizer state live in CUDA memory
- Forward and backward passes run on GPU via custom kernels and cuBLAS
- Training loop, data loading, and tokenization run on CPU (Node.js)
- Loss values and generated tokens are copied back to CPU

## CUDA Kernels

Matrix multiply uses **cuBLAS** (`cublasSgemm` / `cublasSgemmStridedBatched`).

Custom kernels in `native/kernels.cu`:
- `layernorm_kernel` / `layernorm_backward_kernel` — layer normalization
- `attn_scores_kernel` / `batched_attn_scores_kernel` — Q @ K^T with causal mask
- `softmax_kernel` / `softmax_backward_kernel` / `batched_softmax_backward_kernel` — per-row softmax
- `log_softmax_kernel` / `log_softmax_backward_kernel` — output layer
- `tanh_kernel` / `tanh_backward_kernel` — activation
- `gather_kernel` / `gather_backward_kernel` — embedding lookup
- `concat_heads_kernel` / `split_heads_kernel` — multi-head attention concat/split
- `nll_loss_kernel` / `nll_loss_backward_kernel` — loss computation
- `adamw_kernel` — AdamW parameter updates with bias correction
- `add_kernel` / `add_bias_backward_kernel` — element-wise add with broadcast
- `transpose_kernel` / `batched_transpose_kernel` — matrix transpose
- `copy_kernel`, `add_inplace_kernel`, `zero_kernel` — utilities

## GPU architecture

Pass your GPU's compute capability to the build script (defaults to `sm_89`):

```bash
build.cmd sm_86        # Windows
bash build.sh sm_86    # Linux (untested)
```

Common values:
- RTX 4080/4090: `sm_89`
- RTX 3080/3090: `sm_86`
- RTX 2080: `sm_75`
- GTX 1080: `sm_61`
