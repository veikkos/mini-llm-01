// N-API addon that bridges Node.js to CUDA kernels.
//
// GPU tensors live as opaque pointers — JS never touches the GPU memory directly.
// Operations happen entirely on GPU. We only copy data to/from GPU for:
//   - Initial weight setup (CPU → GPU)
//   - Reading loss values (GPU → CPU)
//   - Generation (reading output probabilities)
//   - Saving/loading weights

#include <node_api.h>
#include <cstring>
#include <cstdlib>

// Forward declarations for CUDA functions (defined in kernels.cu)
extern "C" {
    float* cuda_alloc(int size);
    int* cuda_alloc_int(int size);
    void cuda_free(void* ptr);
    void cuda_copy_to_gpu(float* dst, const float* src, int size);
    void cuda_copy_to_gpu_int(int* dst, const int* src, int size);
    void cuda_copy_to_cpu(float* dst, const float* src, int size);
    void cuda_zero(float* ptr, int size);
    void cuda_matmul(const float* A, const float* B, float* C, int M, int K, int N);
    void cuda_matmul_add(const float* A, const float* B, float* C, int M, int K, int N);
    void cuda_transpose(const float* A, float* B, int rows, int cols);
    void cuda_add(const float* A, const float* B, float* C, int size, int cols, int b_size);
    void cuda_add_bias_backward(const float* grad, float* bias_grad, int rows, int cols);
    void cuda_gelu(const float* in, float* out, int size);
    void cuda_gelu_backward(const float* input, const float* grad_out, float* grad_in, int size);
    void cuda_attn_scores(const float* Q, const float* K, float* scores, int T, int D, float scale, int is_causal);
    void cuda_softmax(float* data, int rows, int cols);
    void cuda_softmax_backward(const float* weights, const float* dWeights, float* dScores, int T, float scale, int is_causal);
    void cuda_log_softmax(const float* in, float* out, int rows, int cols);
    void cuda_log_softmax_backward(const float* ls_out, const float* grad_out, float* grad_in, int rows, int cols);
    void cuda_nll_loss(const float* log_probs, const int* targets, float* loss, int T, int vocab_size);
    void cuda_nll_loss_backward(float* grad, const int* targets, int T, int vocab_size, float scale);
    void cuda_gather(const float* table, const int* indices, float* out, int T, int cols);
    void cuda_gather_backward(float* table_grad, const int* indices, const float* out_grad, int T, int cols);
    void cuda_copy(const float* src, float* dst, int size);
    void cuda_add_inplace(const float* src, float* dst, int size);
    void cuda_concat_heads(const float** head_ptrs, float* out, int T, int headDim, int numHeads);
    void cuda_split_heads(const float* in, float** head_ptrs, int T, int headDim, int numHeads);
    const char* cuda_sync();
    void cuda_batched_attn_scores(const float* Q, const float* K, float* scores,
                                   int B, int T, int D, float scale, int is_causal);
    void cuda_batched_softmax_backward(const float* weights, const float* dWeights,
                                        float* dScores, int B, int T, float scale, int is_causal);
    void cuda_batched_matmul(const float* A, const float* B, float* C,
                              int batchCount, int M, int K, int N,
                              long long strideA, long long strideB, long long strideC);
    void cuda_batched_matmul_add(const float* A, const float* B, float* C,
                                  int batchCount, int M, int K, int N,
                                  long long strideA, long long strideB, long long strideC);
    void cuda_batched_transpose(const float* A, float* B, int batchCount, int rows, int cols);

    // Model API (defined in model.cu)
    struct CudaModel;
    CudaModel* model_create(int vocabSize, int embedDim, int contextLen, int numLayers, int numHeads);
    void model_set_param(CudaModel* m, int index, float* data, float* grad, int size);
    float model_forward(CudaModel* m, const int* tokens, const int* targets, int B, int T);
    void model_forward_logprobs(CudaModel* m, const int* tokens, int B, int T);
    void model_backward(CudaModel* m, int B, int T);
    void model_zero_grad(CudaModel* m);
    void model_update(CudaModel* m, float lr);
    float model_train_step(CudaModel* m, const int* all_tokens, int tokenCount,
                            const int* offsets, int B, int T, float lr);
    int model_generate(CudaModel* m, const int* seedTokens, int seedLen,
                        int numTokens, float temperature, int* out_tokens,
                        int (*callback)(int token, void* user), void* user);
    void model_free(CudaModel* m);
}

// BPE API (defined in bpe.cpp)
#include "bpe.h"

// Helper: get pointer from external (stored as bigint in JS)
static float* get_ptr(napi_env env, napi_value val) {
    int64_t addr;
    napi_get_value_int64(env, val, &addr);
    return (float*)(uintptr_t)addr;
}

static int* get_int_ptr(napi_env env, napi_value val) {
    int64_t addr;
    napi_get_value_int64(env, val, &addr);
    return (int*)(uintptr_t)addr;
}

static int get_int(napi_env env, napi_value val) {
    int32_t v;
    napi_get_value_int32(env, val, &v);
    return v;
}

static float get_float(napi_env env, napi_value val) {
    double v;
    napi_get_value_double(env, val, &v);
    return (float)v;
}

static int64_t get_int64(napi_env env, napi_value val) {
    int64_t v;
    napi_get_value_int64(env, val, &v);
    return v;
}

static napi_value make_ptr(napi_env env, void* ptr) {
    napi_value val;
    napi_create_int64(env, (int64_t)(uintptr_t)ptr, &val);
    return val;
}

static napi_value make_undef(napi_env env) {
    napi_value val;
    napi_get_undefined(env, &val);
    return val;
}

// ---- N-API functions ----

// alloc(size) → ptr
static napi_value Alloc(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    return make_ptr(env, cuda_alloc(get_int(env, args[0])));
}

// allocInt(size) → ptr
static napi_value AllocInt(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    return make_ptr(env, cuda_alloc_int(get_int(env, args[0])));
}

// free(ptr)
static napi_value Free(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_free(get_ptr(env, args[0]));
    return make_undef(env);
}

// toGPU(gpu_ptr, float32array)
static napi_value ToGPU(napi_env env, napi_callback_info info) {
    size_t argc = 2; napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    float* gpu = get_ptr(env, args[0]);
    float* cpu; size_t len;
    napi_get_typedarray_info(env, args[1], NULL, &len, (void**)&cpu, NULL, NULL);
    cuda_copy_to_gpu(gpu, cpu, len);
    return make_undef(env);
}

// toGPUInt(gpu_ptr, int32array)
static napi_value ToGPUInt(napi_env env, napi_callback_info info) {
    size_t argc = 2; napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    int* gpu = get_int_ptr(env, args[0]);
    int* cpu; size_t len;
    napi_get_typedarray_info(env, args[1], NULL, &len, (void**)&cpu, NULL, NULL);
    cuda_copy_to_gpu_int(gpu, cpu, len);
    return make_undef(env);
}

// toCPU(float32array, gpu_ptr, size)
static napi_value ToCPU(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    float* cpu; size_t len;
    napi_get_typedarray_info(env, args[0], NULL, &len, (void**)&cpu, NULL, NULL);
    float* gpu = get_ptr(env, args[1]);
    int size = get_int(env, args[2]);
    cuda_copy_to_cpu(cpu, gpu, size);
    return make_undef(env);
}

// zero(ptr, size)
static napi_value Zero(napi_env env, napi_callback_info info) {
    size_t argc = 2; napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_zero(get_ptr(env, args[0]), get_int(env, args[1]));
    return make_undef(env);
}

// matmul(A, B, C, M, K, N)
static napi_value Matmul(napi_env env, napi_callback_info info) {
    size_t argc = 6; napi_value args[6];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_matmul(get_ptr(env, args[0]), get_ptr(env, args[1]), get_ptr(env, args[2]),
                get_int(env, args[3]), get_int(env, args[4]), get_int(env, args[5]));
    return make_undef(env);
}

// matmulAdd(A, B, C, M, K, N) — C += A @ B
static napi_value MatmulAdd(napi_env env, napi_callback_info info) {
    size_t argc = 6; napi_value args[6];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_matmul_add(get_ptr(env, args[0]), get_ptr(env, args[1]), get_ptr(env, args[2]),
                    get_int(env, args[3]), get_int(env, args[4]), get_int(env, args[5]));
    return make_undef(env);
}

// transpose(A, B, rows, cols)
static napi_value Transpose(napi_env env, napi_callback_info info) {
    size_t argc = 4; napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_transpose(get_ptr(env, args[0]), get_ptr(env, args[1]),
                   get_int(env, args[2]), get_int(env, args[3]));
    return make_undef(env);
}

// add(A, B, C, size, cols, b_size)
static napi_value Add(napi_env env, napi_callback_info info) {
    size_t argc = 6; napi_value args[6];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_add(get_ptr(env, args[0]), get_ptr(env, args[1]), get_ptr(env, args[2]),
             get_int(env, args[3]), get_int(env, args[4]), get_int(env, args[5]));
    return make_undef(env);
}

// addBiasBackward(grad, bias_grad, rows, cols)
static napi_value AddBiasBackward(napi_env env, napi_callback_info info) {
    size_t argc = 4; napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_add_bias_backward(get_ptr(env, args[0]), get_ptr(env, args[1]),
                           get_int(env, args[2]), get_int(env, args[3]));
    return make_undef(env);
}

// gelu(in, out, size)
static napi_value Gelu(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_gelu(get_ptr(env, args[0]), get_ptr(env, args[1]), get_int(env, args[2]));
    return make_undef(env);
}

// geluBackward(input, grad_out, grad_in, size)
static napi_value GeluBackward(napi_env env, napi_callback_info info) {
    size_t argc = 4; napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_gelu_backward(get_ptr(env, args[0]), get_ptr(env, args[1]),
                       get_ptr(env, args[2]), get_int(env, args[3]));
    return make_undef(env);
}

// attnScores(Q, K, scores, T, D, scale, is_causal)
static napi_value AttnScores(napi_env env, napi_callback_info info) {
    size_t argc = 7; napi_value args[7];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_attn_scores(get_ptr(env, args[0]), get_ptr(env, args[1]), get_ptr(env, args[2]),
                     get_int(env, args[3]), get_int(env, args[4]),
                     get_float(env, args[5]), get_int(env, args[6]));
    return make_undef(env);
}

// softmax(data, rows, cols)
static napi_value Softmax(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_softmax(get_ptr(env, args[0]), get_int(env, args[1]), get_int(env, args[2]));
    return make_undef(env);
}

// softmaxBackward(weights, dWeights, dScores, T, scale, is_causal)
static napi_value SoftmaxBackward(napi_env env, napi_callback_info info) {
    size_t argc = 6; napi_value args[6];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_softmax_backward(get_ptr(env, args[0]), get_ptr(env, args[1]),
                          get_ptr(env, args[2]), get_int(env, args[3]),
                          get_float(env, args[4]), get_int(env, args[5]));
    return make_undef(env);
}

// logSoftmax(in, out, rows, cols)
static napi_value LogSoftmax(napi_env env, napi_callback_info info) {
    size_t argc = 4; napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_log_softmax(get_ptr(env, args[0]), get_ptr(env, args[1]),
                     get_int(env, args[2]), get_int(env, args[3]));
    return make_undef(env);
}

// logSoftmaxBackward(ls_out, grad_out, grad_in, rows, cols)
static napi_value LogSoftmaxBackward(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_log_softmax_backward(get_ptr(env, args[0]), get_ptr(env, args[1]),
                              get_ptr(env, args[2]), get_int(env, args[3]),
                              get_int(env, args[4]));
    return make_undef(env);
}

// nllLoss(log_probs, targets, loss_ptr, T, vocab_size)
static napi_value NllLoss(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_nll_loss(get_ptr(env, args[0]), get_int_ptr(env, args[1]),
                  get_ptr(env, args[2]), get_int(env, args[3]), get_int(env, args[4]));
    return make_undef(env);
}

// nllLossBackward(grad, targets, T, vocab_size, scale)
static napi_value NllLossBackward(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_nll_loss_backward(get_ptr(env, args[0]), get_int_ptr(env, args[1]),
                           get_int(env, args[2]), get_int(env, args[3]),
                           get_float(env, args[4]));
    return make_undef(env);
}

// gather(table, indices, out, T, cols)
static napi_value Gather(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_gather(get_ptr(env, args[0]), get_int_ptr(env, args[1]),
                get_ptr(env, args[2]), get_int(env, args[3]), get_int(env, args[4]));
    return make_undef(env);
}

// gatherBackward(table_grad, indices, out_grad, T, cols)
static napi_value GatherBackward(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_gather_backward(get_ptr(env, args[0]), get_int_ptr(env, args[1]),
                         get_ptr(env, args[2]), get_int(env, args[3]),
                         get_int(env, args[4]));
    return make_undef(env);
}

// copy(src, dst, size)
static napi_value Copy(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_copy(get_ptr(env, args[0]), get_ptr(env, args[1]), get_int(env, args[2]));
    return make_undef(env);
}

// addInplace(src, dst, size) — dst += src
static napi_value AddInplace(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_add_inplace(get_ptr(env, args[0]), get_ptr(env, args[1]), get_int(env, args[2]));
    return make_undef(env);
}

// concatHeads(headPtrArray, out, T, headDim, numHeads)
// headPtrArray is a JS array of BigInt GPU pointers
static napi_value ConcatHeads(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    float* out = get_ptr(env, args[1]);
    int T = get_int(env, args[2]);
    int headDim = get_int(env, args[3]);
    int numHeads = get_int(env, args[4]);

    // Extract GPU pointers from JS array
    uint32_t len;
    napi_get_array_length(env, args[0], &len);
    const float** ptrs = (const float**)malloc(len * sizeof(float*));
    for (uint32_t i = 0; i < len; i++) {
        napi_value elem;
        napi_get_element(env, args[0], i, &elem);
        ptrs[i] = get_ptr(env, elem);
    }

    cuda_concat_heads(ptrs, out, T, headDim, numHeads);
    free(ptrs);
    return make_undef(env);
}

// splitHeads(in, headPtrArray, T, headDim, numHeads)
static napi_value SplitHeads(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    const float* in = get_ptr(env, args[0]);
    int T = get_int(env, args[2]);
    int headDim = get_int(env, args[3]);
    int numHeads = get_int(env, args[4]);

    uint32_t len;
    napi_get_array_length(env, args[1], &len);
    float** ptrs = (float**)malloc(len * sizeof(float*));
    for (uint32_t i = 0; i < len; i++) {
        napi_value elem;
        napi_get_element(env, args[1], i, &elem);
        ptrs[i] = get_ptr(env, elem);
    }

    cuda_split_heads(in, ptrs, T, headDim, numHeads);
    free(ptrs);
    return make_undef(env);
}

// batchedAttnScores(Q, K, scores, B, T, D, scale, is_causal)
static napi_value BatchedAttnScores(napi_env env, napi_callback_info info) {
    size_t argc = 8; napi_value args[8];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_batched_attn_scores(get_ptr(env, args[0]), get_ptr(env, args[1]), get_ptr(env, args[2]),
                              get_int(env, args[3]), get_int(env, args[4]),
                              get_int(env, args[5]), get_float(env, args[6]), get_int(env, args[7]));
    return make_undef(env);
}

// batchedSoftmaxBackward(weights, dWeights, dScores, B, T, scale, is_causal)
static napi_value BatchedSoftmaxBackward(napi_env env, napi_callback_info info) {
    size_t argc = 7; napi_value args[7];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_batched_softmax_backward(get_ptr(env, args[0]), get_ptr(env, args[1]),
                                   get_ptr(env, args[2]), get_int(env, args[3]),
                                   get_int(env, args[4]), get_float(env, args[5]),
                                   get_int(env, args[6]));
    return make_undef(env);
}

// batchedMatmul(A, B, C, batchCount, M, K, N, strideA, strideB, strideC)
static napi_value BatchedMatmul(napi_env env, napi_callback_info info) {
    size_t argc = 10; napi_value args[10];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_batched_matmul(get_ptr(env, args[0]), get_ptr(env, args[1]), get_ptr(env, args[2]),
                         get_int(env, args[3]), get_int(env, args[4]),
                         get_int(env, args[5]), get_int(env, args[6]),
                         get_int64(env, args[7]), get_int64(env, args[8]), get_int64(env, args[9]));
    return make_undef(env);
}

// batchedMatmulAdd(A, B, C, batchCount, M, K, N, strideA, strideB, strideC)
static napi_value BatchedMatmulAdd(napi_env env, napi_callback_info info) {
    size_t argc = 10; napi_value args[10];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_batched_matmul_add(get_ptr(env, args[0]), get_ptr(env, args[1]), get_ptr(env, args[2]),
                             get_int(env, args[3]), get_int(env, args[4]),
                             get_int(env, args[5]), get_int(env, args[6]),
                             get_int64(env, args[7]), get_int64(env, args[8]), get_int64(env, args[9]));
    return make_undef(env);
}

// batchedTranspose(A, B, batchCount, rows, cols)
static napi_value BatchedTranspose(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    cuda_batched_transpose(get_ptr(env, args[0]), get_ptr(env, args[1]),
                            get_int(env, args[2]), get_int(env, args[3]), get_int(env, args[4]));
    return make_undef(env);
}

// sync() — returns error string or undefined
static napi_value Sync(napi_env env, napi_callback_info info) {
    const char* err = cuda_sync();
    if (err) {
        napi_value result;
        napi_create_string_utf8(env, err, NAPI_AUTO_LENGTH, &result);
        return result;
    }
    return make_undef(env);
}

// --- Model API wrappers ---

// modelCreate(vocabSize, embedDim, contextLen, numLayers, numHeads) → handle
static napi_value ModelCreate(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = model_create(get_int(env, args[0]), get_int(env, args[1]),
                                 get_int(env, args[2]), get_int(env, args[3]),
                                 get_int(env, args[4]));
    return make_ptr(env, m);
}

// modelSetParam(handle, index, data_ptr, grad_ptr, size)
static napi_value ModelSetParam(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = (CudaModel*)(uintptr_t)get_int64(env, args[0]);
    model_set_param(m, get_int(env, args[1]), get_ptr(env, args[2]),
                    get_ptr(env, args[3]), get_int(env, args[4]));
    return make_undef(env);
}

// modelForward(handle, int32tokens, int32targets, B, T) → loss
static napi_value ModelForward(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = (CudaModel*)(uintptr_t)get_int64(env, args[0]);
    int* tokens; size_t tlen;
    napi_get_typedarray_info(env, args[1], NULL, &tlen, (void**)&tokens, NULL, NULL);
    int* targets; size_t tglen;
    napi_get_typedarray_info(env, args[2], NULL, &tglen, (void**)&targets, NULL, NULL);
    int B = get_int(env, args[3]);
    int T = get_int(env, args[4]);
    float loss = model_forward(m, tokens, targets, B, T);
    napi_value result;
    napi_create_double(env, loss, &result);
    return result;
}

// modelBackward(handle, B, T)
static napi_value ModelBackward(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = (CudaModel*)(uintptr_t)get_int64(env, args[0]);
    model_backward(m, get_int(env, args[1]), get_int(env, args[2]));
    return make_undef(env);
}

// modelZeroGrad(handle)
static napi_value ModelZeroGrad(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = (CudaModel*)(uintptr_t)get_int64(env, args[0]);
    model_zero_grad(m);
    return make_undef(env);
}

// modelUpdate(handle, lr)
static napi_value ModelUpdate(napi_env env, napi_callback_info info) {
    size_t argc = 2; napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = (CudaModel*)(uintptr_t)get_int64(env, args[0]);
    model_update(m, get_float(env, args[1]));
    return make_undef(env);
}

// modelTrainStep(handle, int32allTokens, int32offsets, B, T, lr) → float loss
static napi_value ModelTrainStep(napi_env env, napi_callback_info info) {
    size_t argc = 6; napi_value args[6];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = (CudaModel*)(uintptr_t)get_int64(env, args[0]);
    int* allTokens; size_t tokenCount;
    napi_get_typedarray_info(env, args[1], NULL, &tokenCount, (void**)&allTokens, NULL, NULL);
    int* offsets; size_t offsetCount;
    napi_get_typedarray_info(env, args[2], NULL, &offsetCount, (void**)&offsets, NULL, NULL);
    int B = get_int(env, args[3]);
    int T = get_int(env, args[4]);
    float lr = get_float(env, args[5]);

    float loss = model_train_step(m, allTokens, (int)tokenCount, offsets, B, T, lr);

    napi_value result;
    napi_create_double(env, loss, &result);
    return result;
}

// modelGenerate(handle, int32seed, numTokens, temperature) → Int32Array
static napi_value ModelGenerate(napi_env env, napi_callback_info info) {
    size_t argc = 4; napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = (CudaModel*)(uintptr_t)get_int64(env, args[0]);
    int* seed; size_t seedLen;
    napi_get_typedarray_info(env, args[1], NULL, &seedLen, (void**)&seed, NULL, NULL);
    int numTokens = get_int(env, args[2]);
    float temperature = get_float(env, args[3]);

    int maxLen = (int)seedLen + numTokens;
    int* outBuf = (int*)malloc(maxLen * sizeof(int));
    int totalLen = model_generate(m, seed, (int)seedLen, numTokens, temperature, outBuf, NULL, NULL);

    // Create Int32Array result
    napi_value arrayBuffer, result;
    void* data;
    napi_create_arraybuffer(env, totalLen * sizeof(int), &data, &arrayBuffer);
    memcpy(data, outBuf, totalLen * sizeof(int));
    napi_create_typedarray(env, napi_int32_array, totalLen, arrayBuffer, 0, &result);
    free(outBuf);
    return result;
}

// Streaming callback state
struct StreamCtx {
    napi_env env;
    napi_ref callbackRef;
    bool stopped;
};

static int stream_callback(int token, void* user) {
    StreamCtx* ctx = (StreamCtx*)user;
    if (ctx->stopped) return 0;

    napi_value callback, global, argv, result;
    napi_get_reference_value(ctx->env, ctx->callbackRef, &callback);
    napi_get_global(ctx->env, &global);
    napi_create_int32(ctx->env, token, &argv);
    napi_status status = napi_call_function(ctx->env, global, callback, 1, &argv, &result);
    if (status != napi_ok) { ctx->stopped = true; return 0; }

    // If callback returns false, stop
    bool shouldContinue = true;
    napi_get_value_bool(ctx->env, result, &shouldContinue);
    if (!shouldContinue) { ctx->stopped = true; return 0; }
    return 1;
}

// modelGenerateStream(handle, int32seed, numTokens, temperature, callback) → Int32Array
static napi_value ModelGenerateStream(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = (CudaModel*)(uintptr_t)get_int64(env, args[0]);
    int* seed; size_t seedLen;
    napi_get_typedarray_info(env, args[1], NULL, &seedLen, (void**)&seed, NULL, NULL);
    int numTokens = get_int(env, args[2]);
    float temperature = get_float(env, args[3]);

    StreamCtx ctx;
    ctx.env = env;
    ctx.stopped = false;
    napi_create_reference(env, args[4], 1, &ctx.callbackRef);

    int maxLen = (int)seedLen + numTokens;
    int* outBuf = (int*)malloc(maxLen * sizeof(int));
    int totalLen = model_generate(m, seed, (int)seedLen, numTokens, temperature, outBuf, stream_callback, &ctx);

    napi_delete_reference(env, ctx.callbackRef);

    napi_value arrayBuffer, result;
    void* data;
    napi_create_arraybuffer(env, totalLen * sizeof(int), &data, &arrayBuffer);
    memcpy(data, outBuf, totalLen * sizeof(int));
    napi_create_typedarray(env, napi_int32_array, totalLen, arrayBuffer, 0, &result);
    free(outBuf);
    return result;
}

// modelFree(handle)
static napi_value ModelFree(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    CudaModel* m = (CudaModel*)(uintptr_t)get_int64(env, args[0]);
    model_free(m);
    return make_undef(env);
}

// --- BPE API wrappers ---

// bpeCreate() → handle
static napi_value BpeCreate(napi_env env, napi_callback_info info) {
    BPE* bpe = new BPE();
    return make_ptr(env, bpe);
}

// bpeFree(handle)
static napi_value BpeFree(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    BPE* bpe = (BPE*)(uintptr_t)get_int64(env, args[0]);
    delete bpe;
    return make_undef(env);
}

// bpeTrain(handle, string, vocabSize)
static napi_value BpeTrain(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    BPE* bpe = (BPE*)(uintptr_t)get_int64(env, args[0]);

    // Get string data
    size_t strLen;
    napi_get_value_string_utf8(env, args[1], NULL, 0, &strLen);
    std::string text(strLen, '\0');
    napi_get_value_string_utf8(env, args[1], &text[0], strLen + 1, &strLen);

    int vocabSize = get_int(env, args[2]);
    bpe->train(reinterpret_cast<const uint8_t*>(text.data()), text.size(), vocabSize);
    return make_undef(env);
}

// bpeEncode(handle, string) → Int32Array
static napi_value BpeEncode(napi_env env, napi_callback_info info) {
    size_t argc = 2; napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    BPE* bpe = (BPE*)(uintptr_t)get_int64(env, args[0]);

    size_t strLen;
    napi_get_value_string_utf8(env, args[1], NULL, 0, &strLen);
    std::string text(strLen, '\0');
    napi_get_value_string_utf8(env, args[1], &text[0], strLen + 1, &strLen);

    std::vector<int> tokens = bpe->encode(text);

    napi_value arrayBuffer, result;
    void* data;
    napi_create_arraybuffer(env, tokens.size() * sizeof(int), &data, &arrayBuffer);
    memcpy(data, tokens.data(), tokens.size() * sizeof(int));
    napi_create_typedarray(env, napi_int32_array, tokens.size(), arrayBuffer, 0, &result);
    return result;
}

// bpeDecode(handle, Int32Array) → string
static napi_value BpeDecode(napi_env env, napi_callback_info info) {
    size_t argc = 2; napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    BPE* bpe = (BPE*)(uintptr_t)get_int64(env, args[0]);

    int* tokenData; size_t tokenLen;
    napi_get_typedarray_info(env, args[1], NULL, &tokenLen, (void**)&tokenData, NULL, NULL);

    std::vector<int> tokens(tokenData, tokenData + tokenLen);
    std::string text = bpe->decode(tokens);

    napi_value result;
    napi_create_string_utf8(env, text.c_str(), text.size(), &result);
    return result;
}

// bpeSave(handle, path)
static napi_value BpeSave(napi_env env, napi_callback_info info) {
    size_t argc = 2; napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    BPE* bpe = (BPE*)(uintptr_t)get_int64(env, args[0]);

    size_t pathLen;
    napi_get_value_string_utf8(env, args[1], NULL, 0, &pathLen);
    std::string path(pathLen, '\0');
    napi_get_value_string_utf8(env, args[1], &path[0], pathLen + 1, &pathLen);

    bpe->save(path);
    return make_undef(env);
}

// bpeLoad(handle, path)
static napi_value BpeLoad(napi_env env, napi_callback_info info) {
    size_t argc = 2; napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    BPE* bpe = (BPE*)(uintptr_t)get_int64(env, args[0]);

    size_t pathLen;
    napi_get_value_string_utf8(env, args[1], NULL, 0, &pathLen);
    std::string path(pathLen, '\0');
    napi_get_value_string_utf8(env, args[1], &path[0], pathLen + 1, &pathLen);

    bpe->load(path);
    return make_undef(env);
}

// bpeVocabSize(handle) → int
static napi_value BpeVocabSize(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    BPE* bpe = (BPE*)(uintptr_t)get_int64(env, args[0]);
    napi_value result;
    napi_create_int32(env, bpe->vocabSize(), &result);
    return result;
}

// bpeEosToken() → int
static napi_value BpeEosToken(napi_env env, napi_callback_info info) {
    napi_value result;
    napi_create_int32(env, BPE::EOS_TOKEN, &result);
    return result;
}

#define EXPORT_FN(name, fn) { \
    napi_value _fn; \
    napi_create_function(env, #name, NAPI_AUTO_LENGTH, fn, NULL, &_fn); \
    napi_set_named_property(env, exports, #name, _fn); \
}

static napi_value Init(napi_env env, napi_value exports) {
    EXPORT_FN(alloc, Alloc)
    EXPORT_FN(allocInt, AllocInt)
    EXPORT_FN(free, Free)
    EXPORT_FN(toGPU, ToGPU)
    EXPORT_FN(toGPUInt, ToGPUInt)
    EXPORT_FN(toCPU, ToCPU)
    EXPORT_FN(zero, Zero)
    EXPORT_FN(matmul, Matmul)
    EXPORT_FN(matmulAdd, MatmulAdd)
    EXPORT_FN(transpose, Transpose)
    EXPORT_FN(add, Add)
    EXPORT_FN(addBiasBackward, AddBiasBackward)
    EXPORT_FN(gelu, Gelu)
    EXPORT_FN(geluBackward, GeluBackward)
    EXPORT_FN(attnScores, AttnScores)
    EXPORT_FN(softmax, Softmax)
    EXPORT_FN(softmaxBackward, SoftmaxBackward)
    EXPORT_FN(logSoftmax, LogSoftmax)
    EXPORT_FN(logSoftmaxBackward, LogSoftmaxBackward)
    EXPORT_FN(nllLoss, NllLoss)
    EXPORT_FN(nllLossBackward, NllLossBackward)
    EXPORT_FN(gather, Gather)
    EXPORT_FN(gatherBackward, GatherBackward)
    EXPORT_FN(copy, Copy)
    EXPORT_FN(addInplace, AddInplace)
    EXPORT_FN(concatHeads, ConcatHeads)
    EXPORT_FN(splitHeads, SplitHeads)
    EXPORT_FN(sync, Sync)
    EXPORT_FN(batchedAttnScores, BatchedAttnScores)
    EXPORT_FN(batchedSoftmaxBackward, BatchedSoftmaxBackward)
    EXPORT_FN(batchedMatmul, BatchedMatmul)
    EXPORT_FN(batchedMatmulAdd, BatchedMatmulAdd)
    EXPORT_FN(batchedTranspose, BatchedTranspose)
    EXPORT_FN(modelCreate, ModelCreate)
    EXPORT_FN(modelSetParam, ModelSetParam)
    EXPORT_FN(modelForward, ModelForward)
    EXPORT_FN(modelBackward, ModelBackward)
    EXPORT_FN(modelZeroGrad, ModelZeroGrad)
    EXPORT_FN(modelUpdate, ModelUpdate)
    EXPORT_FN(modelTrainStep, ModelTrainStep)
    EXPORT_FN(modelGenerate, ModelGenerate)
    EXPORT_FN(modelGenerateStream, ModelGenerateStream)
    EXPORT_FN(modelFree, ModelFree)
    EXPORT_FN(bpeCreate, BpeCreate)
    EXPORT_FN(bpeFree, BpeFree)
    EXPORT_FN(bpeTrain, BpeTrain)
    EXPORT_FN(bpeEncode, BpeEncode)
    EXPORT_FN(bpeDecode, BpeDecode)
    EXPORT_FN(bpeSave, BpeSave)
    EXPORT_FN(bpeLoad, BpeLoad)
    EXPORT_FN(bpeVocabSize, BpeVocabSize)
    EXPORT_FN(bpeEosToken, BpeEosToken)
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
