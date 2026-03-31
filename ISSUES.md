# Known Issues

## Bugs

### 1. No bounds checking on batch offsets
**File:** `native/model.cu:648-653`

`model_train_step` reads `all_tokens[start + T]` where `start = offsets[b]`. If the JS side passes a bad offset (or `start + T >= tokenCount`), this reads out of bounds from the CPU array. The JS side (`train.js:85`) does `Math.random() * (tokens.length - contextLen - 1)` which is correct, but the C++ layer is unprotected.

## Performance Issues

### 2. BPE encode is O(merges x sequence_length)
**File:** `native/bpe.cpp:102-133`

`encode()` iterates over all merge rules sequentially, scanning the entire token array each time. For vocab size 4096, that's ~3839 passes. A priority-queue approach was tested but showed no measurable improvement at current data sizes due to heap overhead. May matter for much larger texts or vocabularies.

### 3. layernorm_backward atomic contention (partially addressed)
**File:** `native/kernels.cu`

Per-row work is now parallelized across 256 threads, but each row-block still atomically accumulates into global `dgamma[j]` and `dbeta[j]`. With many rows (large batches), this creates contention on the same memory locations. A two-pass approach (block-local accumulation + separate reduction kernel) would eliminate cross-block atomics entirely.

## Resource Management

### 4. GpuTensor has no destructor/cleanup on the JS side
**File:** `gpu-tensor.js`

`GpuTensor` objects allocate GPU memory via `cuda.alloc()` but there's no destructor, no weak reference callback, and the `.free()` method is never called anywhere in the codebase. The tensors are only cleaned up on process exit.

## Minor Issues

- **`download-tinystories.js:40-41`**: The `fileStream.close()` on error doesn't await the close completing before `unlinkSync`.
- **`encode-worker.js:14`**: Workers call `process.exit(0)` which kills the thread abruptly; returning/closing the port would be cleaner.
