# Mini-LLM-01

*Unsuprisingly created with an AI agent as an educational study. Not production grade or anything but does something. No Python!*

A small transformer language model running entirely on the GPU via custom CUDA kernels.
Includes BPE tokenizer, AdamW optimizer, and TinyStories training pipeline.

## Example output (15,000 steps, 250MB TinyStories subset)

> Once upon a time there was a little girl called Jane. Jane loved to explore the world. She was always busy exploring the woods near her house.
>
> One day, Jane's mum said it was time to go to the park to play. She took out a lot of fun things, it was a track, and soon she was almost got there to catch it. She was so excited and said "Let's go outside and play on it!" She held the track in her hand.
>
> Jane was so excited. She ran outside to watch and saw all the different things she could go there. She wanted to bring something so wrong and her mom gave her a big hug. Jane was so proud of herself for being a great join her new track. She smiled and said "thank you so much!" Jane smiled and said happily.
>
> From then on, she knew she always could do lots of fun things with her favourite things!

## Example output (50,000 steps, 1GB TinyStories, vocab 8192, context 256, ~9M params)

> Once upon a time, there was a girl named Spy. Spy was three years old and liked to spend time with her family.
>
> One day, Spy found a very special present in the garden. It was very big and had a shiny jewel in her hand. She asked her mommy what it was and she replied, "I found something that can help me pick."
>
> Spy smiled and said, "That's a special key!" Her mommy said, "That's very clever, Spy. That's why it can give you a special gift."
>
> Spy learned her lesson and from then on, she and her family enjoyed the special surprise each day.

## Requirements

- NVIDIA GPU (tested on RTX 4080)
- CUDA Toolkit installed (`nvcc` in PATH)
- Visual Studio Developer Command Prompt (Windows)
- Node.js

## Quick Start

```bash
# Build (from Developer Command Prompt, see GPU architecture section below)
build.cmd

# Download TinyStories dataset (~250MB subset, use --size for more)
node download-tinystories.js
# node download-tinystories.js --size 1000  # 1GB

# Train BPE vocabulary (use --maxchars to limit corpus size)
node train-vocab.js --input data/tinystories.txt

# Pre-encode text to binary tokens (one-time)
node encode.js --input data/tinystories.txt

# Train (loads pre-encoded tokens instantly, use --no-gen to skip sample generation)
node train.js --tokens tokens.bin --steps 5000
# node train.js --tokens tokens.bin --steps 50000  # train longer for better quality

# Generate text (interactive)
node generate.js
# Generate text (single output)
node generate.js --seed "The little dog" --temp 0.8 --oneshot
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
