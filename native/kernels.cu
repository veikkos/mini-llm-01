// CUDA kernels for the mini LLM.
//
// Each kernel runs on the GPU — hundreds/thousands of threads in parallel.
// A thread computes one (or a few) output elements.
//
// Key idea: instead of nested for-loops on CPU, we launch a GRID of threads
// where each thread handles one piece of the computation simultaneously.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Global cuBLAS handle — initialized on first use
static cublasHandle_t cublas_handle = nullptr;

static void ensure_cublas() {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
}

// --- Transpose: B = A^T ---
// A is [rows, cols], B is [cols, rows]
__global__ void transpose_kernel(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

// --- Batched Transpose: B matrices of [rows, cols] -> B matrices of [cols, rows] ---
__global__ void batched_transpose_kernel(const float* A, float* B_out,
                                          int batchCount, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchCount * rows * cols;
    if (idx >= total) return;

    int b = idx / (rows * cols);
    int rem = idx % (rows * cols);
    int r = rem / cols;
    int c = rem % cols;

    int in_idx = b * rows * cols + r * cols + c;
    int out_idx = b * cols * rows + c * rows + r;
    B_out[out_idx] = A[in_idx];
}

// --- Element-wise Add: C = A + B ---
// With broadcasting: if B has fewer elements, it's a bias (one row repeated)
__global__ void add_kernel(const float* A, const float* B, float* C,
                            int size, int cols, int b_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (b_size == cols) {
            C[i] = A[i] + B[i % cols]; // broadcast bias
        } else {
            C[i] = A[i] + B[i];
        }
    }
}

// --- Element-wise Add Backward for bias ---
// Sum gradients across rows into bias gradient
__global__ void add_bias_backward_kernel(const float* grad, float* bias_grad,
                                          int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int r = 0; r < rows; r++) {
            sum += grad[r * cols + col];
        }
        bias_grad[col] += sum;
    }
}

// --- Tanh ---
__global__ void tanh_kernel(const float* in, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = tanhf(in[i]);
    }
}

// --- Tanh Backward: grad_in += (1 - tanh_out^2) * grad_out ---
__global__ void tanh_backward_kernel(const float* tanh_out, const float* grad_out,
                                      float* grad_in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float t = tanh_out[i];
        grad_in[i] += (1.0f - t * t) * grad_out[i];
    }
}

// --- Layer Normalization ---
// out[i] = gamma * (x[i] - mean) / sqrt(var + eps) + beta
// Each block handles one row (one token position)
// Also saves mean and rstd (1/sqrt(var+eps)) for backward pass
__global__ void layernorm_kernel(const float* x, const float* gamma, const float* beta,
                                  float* out, float* mean_out, float* rstd_out,
                                  int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* xr = x + row * cols;
    float* or_ = out + row * cols;
    float eps = 1e-5f;

    // Mean
    float mean = 0.0f;
    for (int j = 0; j < cols; j++) mean += xr[j];
    mean /= cols;

    // Variance
    float var = 0.0f;
    for (int j = 0; j < cols; j++) {
        float d = xr[j] - mean;
        var += d * d;
    }
    var /= cols;
    float rstd = 1.0f / sqrtf(var + eps);

    // Normalize and scale
    for (int j = 0; j < cols; j++) {
        or_[j] = gamma[j] * (xr[j] - mean) * rstd + beta[j];
    }

    if (mean_out) mean_out[row] = mean;
    if (rstd_out) rstd_out[row] = rstd;
}

// --- Layer Normalization Backward ---
// Computes dx, dgamma, dbeta from dout
__global__ void layernorm_backward_kernel(const float* dout, const float* x,
                                           const float* gamma, const float* mean,
                                           const float* rstd, float* dx,
                                           float* dgamma, float* dbeta,
                                           int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* dr = dout + row * cols;
    const float* xr = x + row * cols;
    float* dxr = dx + row * cols;
    float m = mean[row];
    float rs = rstd[row];

    // Accumulate dgamma and dbeta
    for (int j = 0; j < cols; j++) {
        float xhat = (xr[j] - m) * rs;
        atomicAdd(&dgamma[j], dr[j] * xhat);
        atomicAdd(&dbeta[j], dr[j]);
    }

    // Compute dx
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int j = 0; j < cols; j++) {
        float xhat = (xr[j] - m) * rs;
        sum1 += dr[j] * gamma[j];
        sum2 += dr[j] * gamma[j] * xhat;
    }

    for (int j = 0; j < cols; j++) {
        float xhat = (xr[j] - m) * rs;
        dxr[j] += rs * (dr[j] * gamma[j] - (sum1 + xhat * sum2) / cols);
    }
}

// --- Attention Scores: scores = Q @ K^T * scale, with causal mask ---
// Q is [T, D], K is [T, D], scores is [T, T]
__global__ void attn_scores_kernel(const float* Q, const float* K, float* scores,
                                    int T, int D, float scale, int is_causal) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < T && col < T) {
        if (is_causal && col > row) {
            scores[row * T + col] = -1e9f;
        } else {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += Q[row * D + d] * K[col * D + d];
            }
            scores[row * T + col] = dot * scale;
        }
    }
}

// --- Batched Attention Scores: scores = Q @ K^T * scale, with causal mask ---
// Q is [B*T, D], K is [B*T, D], scores is [B*T, T]
// Each batch's attention is independent — no cross-batch attention
__global__ void batched_attn_scores_kernel(const float* Q, const float* K, float* scores,
                                            int B, int T, int D, float scale, int is_causal) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..B*T-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..T-1

    if (row < B * T && col < T) {
        int b = row / T;
        int local_row = row % T;

        if (is_causal && col > local_row) {
            scores[row * T + col] = -1e9f;
        } else {
            int k_row = b * T + col;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += Q[row * D + d] * K[k_row * D + d];
            }
            scores[row * T + col] = dot * scale;
        }
    }
}

// --- Softmax (per row, in-place) ---
// Each block handles one row
__global__ void softmax_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_data = data + row * cols;

    // Find max
    float max_val = -1e30f;
    for (int j = 0; j < cols; j++) {
        if (row_data[j] > max_val) max_val = row_data[j];
    }

    // Exp and sum
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        row_data[j] = expf(row_data[j] - max_val);
        sum += row_data[j];
    }

    // Normalize
    for (int j = 0; j < cols; j++) {
        row_data[j] /= sum;
    }
}

// --- Log Softmax (per row) ---
__global__ void log_softmax_kernel(const float* in, float* out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = in + row * cols;
    float* out_row = out + row * cols;

    float max_val = -1e30f;
    for (int j = 0; j < cols; j++) {
        if (in_row[j] > max_val) max_val = in_row[j];
    }

    float log_sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        log_sum += expf(in_row[j] - max_val);
    }
    log_sum = max_val + logf(log_sum);

    for (int j = 0; j < cols; j++) {
        out_row[j] = in_row[j] - log_sum;
    }
}

// --- Log Softmax Backward ---
__global__ void log_softmax_backward_kernel(const float* log_softmax_out,
                                             const float* grad_out,
                                             float* grad_in,
                                             int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* ls = log_softmax_out + row * cols;
    const float* go = grad_out + row * cols;
    float* gi = grad_in + row * cols;

    float sum_grad = 0.0f;
    for (int j = 0; j < cols; j++) {
        sum_grad += go[j];
    }

    for (int j = 0; j < cols; j++) {
        gi[j] += go[j] - expf(ls[j]) * sum_grad;
    }
}

// --- NLL Loss ---
// Picks log_prob[i][target[i]] for each row, averages
__global__ void nll_loss_kernel(const float* log_probs, const int* targets,
                                 float* loss, int T, int vocab_size) {
    // Single-thread kernel — tiny operation
    float sum = 0.0f;
    for (int i = 0; i < T; i++) {
        sum -= log_probs[i * vocab_size + targets[i]];
    }
    *loss = sum / T;
}

// --- NLL Loss Backward ---
__global__ void nll_loss_backward_kernel(float* grad, const int* targets,
                                          int T, int vocab_size, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) {
        grad[i * vocab_size + targets[i]] -= scale / T;
    }
}

// --- AdamW Update ---
// Decoupled weight decay: apply weight decay directly to params, not through gradient
__global__ void adamw_kernel(float* param, const float* grad, float* m, float* v,
                              float lr, float beta1, float beta2, float eps,
                              float weight_decay, float bias_corr1, float bias_corr2,
                              int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] / bias_corr1;
        float v_hat = v[i] / bias_corr2;
        param[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param[i]);
    }
}

// --- Zero memory ---
__global__ void zero_kernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = 0.0f;
    }
}

// --- Gather rows (embedding lookup) ---
// out[i] = table[indices[i]] (copies a full row)
__global__ void gather_kernel(const float* table, const int* indices, float* out,
                               int T, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // which index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // which column

    if (i < T && j < cols) {
        out[i * cols + j] = table[indices[i] * cols + j];
    }
}

// --- Gather Backward: scatter-add gradients back to table ---
__global__ void gather_backward_kernel(float* table_grad, const int* indices,
                                        const float* out_grad, int T, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < T && j < cols) {
        atomicAdd(&table_grad[indices[i] * cols + j], out_grad[i * cols + j]);
    }
}

// --- Softmax backward (for attention weights) ---
// Given dWeights and weights, compute dScores
__global__ void softmax_backward_kernel(const float* weights, const float* dWeights,
                                         float* dScores, int T, float scale,
                                         int is_causal) {
    int row = blockIdx.x;
    if (row >= T) return;

    const float* w = weights + row * T;
    const float* dw = dWeights + row * T;
    float* ds = dScores + row * T;

    float dot = 0.0f;
    for (int j = 0; j < T; j++) {
        dot += dw[j] * w[j];
    }

    for (int j = 0; j < T; j++) {
        if (is_causal && j > row) {
            ds[j] = 0.0f;
        } else {
            ds[j] = w[j] * (dw[j] - dot) * scale;
        }
    }
}


// --- Batched Softmax backward ---
__global__ void batched_softmax_backward_kernel(const float* weights, const float* dWeights,
                                                 float* dScores, int B, int T, float scale,
                                                 int is_causal) {
    int row = blockIdx.x;
    if (row >= B * T) return;

    int local_row = row % T;

    const float* w = weights + row * T;
    const float* dw = dWeights + row * T;
    float* ds = dScores + row * T;

    float dot = 0.0f;
    for (int j = 0; j < T; j++) {
        dot += dw[j] * w[j];
    }

    for (int j = 0; j < T; j++) {
        if (is_causal && j > local_row) {
            ds[j] = 0.0f;
        } else {
            ds[j] = w[j] * (dw[j] - dot) * scale;
        }
    }
}


// --- Concat heads: multiple [T, headDim] → one [T, numHeads * headDim] ---
// Each thread copies one element from the right head into the right column
// heads is an array of pointers, but we pass them as a flat buffer with offsets
__global__ void concat_heads_kernel(const float* const* heads, float* out,
                                     int T, int headDim, int numHeads) {
    int t = blockIdx.y * blockDim.y + threadIdx.y;  // row (time step)
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // flat index across all head columns

    int totalCols = headDim * numHeads;
    if (t < T && idx < totalCols) {
        int h = idx / headDim;   // which head
        int j = idx % headDim;   // which column within head
        out[t * totalCols + idx] = heads[h][t * headDim + j];
    }
}

// --- Split (un-concat): one [T, numHeads * headDim] → multiple [T, headDim] ---
__global__ void split_heads_kernel(const float* in, float* const* heads,
                                    int T, int headDim, int numHeads) {
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int totalCols = headDim * numHeads;
    if (t < T && idx < totalCols) {
        int h = idx / headDim;
        int j = idx % headDim;
        heads[h][t * headDim + j] = in[t * totalCols + idx];
    }
}

// --- Copy kernel: dst = src ---
__global__ void copy_kernel(const float* src, float* dst, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dst[i] = src[i];
    }
}

// --- Add in-place: dst += src ---
__global__ void add_inplace_kernel(const float* src, float* dst, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dst[i] += src[i];
    }
}


// =====================================================================
// Host-side wrapper functions (called from the N-API addon)
// =====================================================================

// Helper: launch config
inline dim3 grid2d(int x, int y, int block = 16) {
    return dim3((x + block - 1) / block, (y + block - 1) / block);
}

inline dim3 block2d(int block = 16) {
    return dim3(block, block);
}

inline int grid1d(int n, int block = 256) {
    return (n + block - 1) / block;
}

extern "C" {

// --- GPU memory management ---
float* cuda_alloc(int size) {
    float* ptr;
    cudaMalloc(&ptr, size * sizeof(float));
    return ptr;
}

int* cuda_alloc_int(int size) {
    int* ptr;
    cudaMalloc(&ptr, size * sizeof(int));
    return ptr;
}

void cuda_free(void* ptr) {
    cudaFree(ptr);
}

void cuda_copy_to_gpu(float* dst, const float* src, int size) {
    cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyHostToDevice);
}

void cuda_copy_to_gpu_int(int* dst, const int* src, int size) {
    cudaMemcpy(dst, src, size * sizeof(int), cudaMemcpyHostToDevice);
}

void cuda_copy_to_cpu(float* dst, const float* src, int size) {
    cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void cuda_zero(float* ptr, int size) {
    zero_kernel<<<grid1d(size), 256>>>(ptr, size);
}

// --- Operations ---

// Row-major C = A @ B using cuBLAS (column-major).
// cuBLAS sees col-major matrices, so we compute B^T @ A^T = (A@B)^T in col-major,
// which gives us A@B in row-major.
void cuda_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    ensure_cublas();
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

void cuda_matmul_add(const float* A, const float* B, float* C, int M, int K, int N) {
    ensure_cublas();
    float alpha = 1.0f, beta = 1.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

void cuda_batched_matmul(const float* A, const float* B, float* C,
                          int batchCount, int M, int K, int N,
                          long long strideA, long long strideB, long long strideC) {
    ensure_cublas();
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               B, N, strideB,
                               A, K, strideA,
                               &beta, C, N, strideC,
                               batchCount);
}

void cuda_batched_matmul_add(const float* A, const float* B, float* C,
                              int batchCount, int M, int K, int N,
                              long long strideA, long long strideB, long long strideC) {
    ensure_cublas();
    float alpha = 1.0f, beta = 1.0f;
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               B, N, strideB,
                               A, K, strideA,
                               &beta, C, N, strideC,
                               batchCount);
}

void cuda_transpose(const float* A, float* B, int rows, int cols) {
    transpose_kernel<<<grid2d(cols, rows), block2d()>>>(A, B, rows, cols);
}

void cuda_batched_transpose(const float* A, float* B, int batchCount, int rows, int cols) {
    int total = batchCount * rows * cols;
    batched_transpose_kernel<<<grid1d(total), 256>>>(A, B, batchCount, rows, cols);
}

void cuda_add(const float* A, const float* B, float* C, int size, int cols, int b_size) {
    add_kernel<<<grid1d(size), 256>>>(A, B, C, size, cols, b_size);
}

void cuda_add_bias_backward(const float* grad, float* bias_grad, int rows, int cols) {
    add_bias_backward_kernel<<<grid1d(cols), 256>>>(grad, bias_grad, rows, cols);
}

void cuda_tanh(const float* in, float* out, int size) {
    tanh_kernel<<<grid1d(size), 256>>>(in, out, size);
}

void cuda_tanh_backward(const float* tanh_out, const float* grad_out,
                         float* grad_in, int size) {
    tanh_backward_kernel<<<grid1d(size), 256>>>(tanh_out, grad_out, grad_in, size);
}

void cuda_layernorm(const float* x, const float* gamma, const float* beta,
                     float* out, float* mean, float* rstd, int rows, int cols) {
    layernorm_kernel<<<rows, 1>>>(x, gamma, beta, out, mean, rstd, rows, cols);
}

void cuda_layernorm_backward(const float* dout, const float* x,
                              const float* gamma, const float* mean,
                              const float* rstd, float* dx,
                              float* dgamma, float* dbeta, int rows, int cols) {
    layernorm_backward_kernel<<<rows, 1>>>(dout, x, gamma, mean, rstd, dx, dgamma, dbeta, rows, cols);
}

void cuda_attn_scores(const float* Q, const float* K, float* scores,
                       int T, int D, float scale, int is_causal) {
    attn_scores_kernel<<<grid2d(T, T), block2d()>>>(Q, K, scores, T, D, scale, is_causal);
}

void cuda_batched_attn_scores(const float* Q, const float* K, float* scores,
                               int B, int T, int D, float scale, int is_causal) {
    batched_attn_scores_kernel<<<grid2d(T, B * T), block2d()>>>(Q, K, scores, B, T, D, scale, is_causal);
}

void cuda_softmax(float* data, int rows, int cols) {
    softmax_kernel<<<rows, 1>>>(data, rows, cols);
}

void cuda_softmax_backward(const float* weights, const float* dWeights,
                             float* dScores, int T, float scale, int is_causal) {
    softmax_backward_kernel<<<T, 1>>>(weights, dWeights, dScores, T, scale, is_causal);
}

void cuda_batched_softmax_backward(const float* weights, const float* dWeights,
                                    float* dScores, int B, int T, float scale, int is_causal) {
    batched_softmax_backward_kernel<<<B * T, 1>>>(weights, dWeights, dScores, B, T, scale, is_causal);
}

void cuda_log_softmax(const float* in, float* out, int rows, int cols) {
    log_softmax_kernel<<<rows, 1>>>(in, out, rows, cols);
}

void cuda_log_softmax_backward(const float* ls_out, const float* grad_out,
                                float* grad_in, int rows, int cols) {
    log_softmax_backward_kernel<<<rows, 1>>>(ls_out, grad_out, grad_in, rows, cols);
}

void cuda_nll_loss(const float* log_probs, const int* targets, float* loss,
                    int T, int vocab_size) {
    nll_loss_kernel<<<1, 1>>>(log_probs, targets, loss, T, vocab_size);
}

void cuda_nll_loss_backward(float* grad, const int* targets, int T,
                              int vocab_size, float scale) {
    nll_loss_backward_kernel<<<grid1d(T), 256>>>(grad, targets, T, vocab_size, scale);
}

void cuda_adamw(float* param, const float* grad, float* m, float* v,
                 float lr, float beta1, float beta2, float eps,
                 float weight_decay, int step, int size) {
    float bias_corr1 = 1.0f - powf(beta1, (float)step);
    float bias_corr2 = 1.0f - powf(beta2, (float)step);
    adamw_kernel<<<grid1d(size), 256>>>(param, grad, m, v, lr, beta1, beta2, eps,
                                         weight_decay, bias_corr1, bias_corr2, size);
}

void cuda_gather(const float* table, const int* indices, float* out, int T, int cols) {
    gather_kernel<<<grid2d(cols, T), block2d()>>>(table, indices, out, T, cols);
}

void cuda_gather_backward(float* table_grad, const int* indices,
                            const float* out_grad, int T, int cols) {
    gather_backward_kernel<<<grid2d(cols, T), block2d()>>>(table_grad, indices, out_grad, T, cols);
}

void cuda_copy(const float* src, float* dst, int size) {
    copy_kernel<<<grid1d(size), 256>>>(src, dst, size);
}

void cuda_add_inplace(const float* src, float* dst, int size) {
    add_inplace_kernel<<<grid1d(size), 256>>>(src, dst, size);
}

// Concat heads: takes array of GPU pointers, writes interleaved output
// head_ptrs is a HOST array of device pointers, copied to GPU temporarily
void cuda_concat_heads(const float** head_ptrs, float* out, int T, int headDim, int numHeads) {
    // Upload pointer array to GPU
    const float** d_ptrs;
    cudaMalloc(&d_ptrs, numHeads * sizeof(float*));
    cudaMemcpy((void*)d_ptrs, head_ptrs, numHeads * sizeof(float*), cudaMemcpyHostToDevice);

    int totalCols = headDim * numHeads;
    concat_heads_kernel<<<grid2d(totalCols, T), block2d()>>>(d_ptrs, out, T, headDim, numHeads);

    cudaFree((void*)d_ptrs);
}

// Split (un-concat): writes interleaved input into separate head buffers
void cuda_split_heads(const float* in, float** head_ptrs, int T, int headDim, int numHeads) {
    float** d_ptrs;
    cudaMalloc(&d_ptrs, numHeads * sizeof(float*));
    cudaMemcpy(d_ptrs, head_ptrs, numHeads * sizeof(float*), cudaMemcpyHostToDevice);

    int totalCols = headDim * numHeads;
    split_heads_kernel<<<grid2d(totalCols, T), block2d()>>>( in, (float* const*)d_ptrs, T, headDim, numHeads);

    cudaFree(d_ptrs);
}

const char* cuda_sync() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return cudaGetErrorString(err);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return cudaGetErrorString(err);
    }
    return nullptr;
}

} // extern "C"
