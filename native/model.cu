// C++ model that orchestrates all CUDA kernel calls for forward/backward/update.
// Eliminates hundreds of JS→N-API round trips per training step.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdio>

// Forward declarations for kernels (defined in kernels.cu)
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
    void cuda_batched_matmul(const float* A, const float* B, float* C,
                              int batchCount, int M, int K, int N,
                              long long strideA, long long strideB, long long strideC);
    void cuda_transpose(const float* A, float* B, int rows, int cols);
    void cuda_batched_transpose(const float* A, float* B, int batchCount, int rows, int cols);
    void cuda_add(const float* A, const float* B, float* C, int size, int cols, int b_size);
    void cuda_add_bias_backward(const float* grad, float* bias_grad, int rows, int cols);
    void cuda_tanh(const float* in, float* out, int size);
    void cuda_tanh_backward(const float* tanh_out, const float* grad_out, float* grad_in, int size);
    void cuda_batched_attn_scores(const float* Q, const float* K, float* scores,
                                   int B, int T, int D, float scale, int is_causal);
    void cuda_softmax(float* data, int rows, int cols);
    void cuda_batched_softmax_backward(const float* weights, const float* dWeights,
                                        float* dScores, int B, int T, float scale, int is_causal);
    void cuda_log_softmax(const float* in, float* out, int rows, int cols);
    void cuda_log_softmax_backward(const float* ls_out, const float* grad_out,
                                    float* grad_in, int rows, int cols);
    void cuda_nll_loss(const float* log_probs, const int* targets, float* loss, int T, int vocab_size);
    void cuda_nll_loss_backward(float* grad, const int* targets, int T, int vocab_size, float scale);
    void cuda_adamw(float* param, const float* grad, float* m, float* v,
                     float lr, float beta1, float beta2, float eps,
                     float weight_decay, int step, int size);
    void cuda_gather(const float* table, const int* indices, float* out, int T, int cols);
    void cuda_gather_backward(float* table_grad, const int* indices, const float* out_grad, int T, int cols);
    void cuda_copy(const float* src, float* dst, int size);
    void cuda_add_inplace(const float* src, float* dst, int size);
    void cuda_concat_heads(const float** head_ptrs, float* out, int T, int headDim, int numHeads);
    void cuda_split_heads(const float* in, float** head_ptrs, int T, int headDim, int numHeads);
    void cuda_layernorm(const float* x, const float* gamma, const float* beta,
                         float* out, float* mean, float* rstd, int rows, int cols);
    void cuda_layernorm_backward(const float* dout, const float* x,
                                  const float* gamma, const float* mean,
                                  const float* rstd, float* dx,
                                  float* dgamma, float* dbeta, int rows, int cols);
    const char* cuda_sync();
}

// --- Model structure ---

struct Param {
    float* data;
    float* grad;
    int size;
};

struct LayerBuffers {
    // Layer norm outputs + stats
    float* ln1Out;       // pre-attention norm output
    float* ln1Mean;
    float* ln1Rstd;
    float* ln2Out;       // pre-FF norm output
    float* ln2Mean;
    float* ln2Rstd;
    // Per-head (arrays of numHeads pointers)
    float** headQ;
    float** headK;
    float** headV;
    float** headScores;
    float** headWeights;
    float** headOut;
    // Concatenated + projected
    float* concat;
    float* attnOut;
    float* afterAttn;
    // FF
    float* ff1;
    float* ff1Act;
    float* ff2;
    float* afterFF;
    // Backward
    float* dLn1Out;
    float* dLn2Out;
    float** dHeadOut;
    float** dHeadWeights;
    float** dHeadScores;
    float** dHeadQ;
    float** dHeadK;
    float** dHeadV;
    float* dConcat;
    float* dFF1Act;
    float* dFF1;
    // Scratch
    float* tmp;
};

struct ModelBuffers {
    float* tokEmb;
    float* posEmb;
    float* x;
    float** layerInputs;
    float* logits;
    float* logProbs;
    float* loss;
    float* dLogProbs;
    float* dLogits;
    float* dX;
    float* tmp;
    int* tokens;
    int* targets;
    int* positions;
};

struct CudaModel {
    // Config
    int vocabSize, embedDim, contextLen, numLayers, numHeads, headDim, ffDim;

    // Parameters
    int numParams;
    Param* params;

    // Parameter indices
    int iTokenEmbed, iPosEmbed, iOutW, iOutB;
    // Per layer: starts at iLayerBase + layer * paramsPerLayer
    int iLayerBase, paramsPerLayer;
    // Within layer: head[h] Wq=h*3, Wk=h*3+1, Wv=h*3+2, then Wo, ff1W, ff1B, ff2W, ff2B

    // Buffers
    LayerBuffers* layerBufs;
    ModelBuffers bufs;
    int allocB, allocT;

    // Position array cache
    int* posCache;
    int posCacheSize;
    int posCacheT;

    // AdamW optimizer state
    float** adam_m;  // first moment (per param)
    float** adam_v;  // second moment (per param)
    int adamStep;    // timestep counter
};

static float** alloc_ptr_array(int n) {
    return (float**)malloc(n * sizeof(float*));
}

static void alloc_layer_bufs(LayerBuffers* lb, int numHeads, int BT, int T, int d, int E, int ffDim) {
    lb->ln1Out = cuda_alloc(BT * E);
    lb->ln1Mean = cuda_alloc(BT);
    lb->ln1Rstd = cuda_alloc(BT);
    lb->ln2Out = cuda_alloc(BT * E);
    lb->ln2Mean = cuda_alloc(BT);
    lb->ln2Rstd = cuda_alloc(BT);
    lb->dLn1Out = cuda_alloc(BT * E);
    lb->dLn2Out = cuda_alloc(BT * E);

    lb->headQ = alloc_ptr_array(numHeads);
    lb->headK = alloc_ptr_array(numHeads);
    lb->headV = alloc_ptr_array(numHeads);
    lb->headScores = alloc_ptr_array(numHeads);
    lb->headWeights = alloc_ptr_array(numHeads);
    lb->headOut = alloc_ptr_array(numHeads);
    lb->dHeadOut = alloc_ptr_array(numHeads);
    lb->dHeadWeights = alloc_ptr_array(numHeads);
    lb->dHeadScores = alloc_ptr_array(numHeads);
    lb->dHeadQ = alloc_ptr_array(numHeads);
    lb->dHeadK = alloc_ptr_array(numHeads);
    lb->dHeadV = alloc_ptr_array(numHeads);

    for (int h = 0; h < numHeads; h++) {
        lb->headQ[h] = cuda_alloc(BT * d);
        lb->headK[h] = cuda_alloc(BT * d);
        lb->headV[h] = cuda_alloc(BT * d);
        lb->headScores[h] = cuda_alloc(BT * T);
        lb->headWeights[h] = cuda_alloc(BT * T);
        lb->headOut[h] = cuda_alloc(BT * d);
        lb->dHeadOut[h] = cuda_alloc(BT * d);
        lb->dHeadWeights[h] = cuda_alloc(BT * T);
        lb->dHeadScores[h] = cuda_alloc(BT * T);
        lb->dHeadQ[h] = cuda_alloc(BT * d);
        lb->dHeadK[h] = cuda_alloc(BT * d);
        lb->dHeadV[h] = cuda_alloc(BT * d);
    }

    lb->concat = cuda_alloc(BT * E);
    lb->attnOut = cuda_alloc(BT * E);
    lb->afterAttn = cuda_alloc(BT * E);
    lb->ff1 = cuda_alloc(BT * ffDim);
    lb->ff1Act = cuda_alloc(BT * ffDim);
    lb->ff2 = cuda_alloc(BT * E);
    lb->afterFF = cuda_alloc(BT * E);
    lb->dConcat = cuda_alloc(BT * E);
    lb->dFF1Act = cuda_alloc(BT * ffDim);
    lb->dFF1 = cuda_alloc(BT * ffDim);

    int tmpSize = BT * E;
    if (BT * T > tmpSize) tmpSize = BT * T;
    if (BT * ffDim > tmpSize) tmpSize = BT * ffDim;
    if (E * ffDim > tmpSize) tmpSize = E * ffDim;
    lb->tmp = cuda_alloc(tmpSize);
}

// --- Param indexing helpers ---

static inline Param& layer_param(CudaModel* m, int layer, int offset) {
    return m->params[m->iLayerBase + layer * m->paramsPerLayer + offset];
}

static inline Param& head_Wq(CudaModel* m, int layer, int h) {
    return layer_param(m, layer, h * 3);
}
static inline Param& head_Wk(CudaModel* m, int layer, int h) {
    return layer_param(m, layer, h * 3 + 1);
}
static inline Param& head_Wv(CudaModel* m, int layer, int h) {
    return layer_param(m, layer, h * 3 + 2);
}
// Per-layer param layout: head[0..H-1] × (Wq,Wk,Wv), Wo, ff1W, ff1B, ff2W, ff2B, ln1G, ln1B, ln2G, ln2B
static inline Param& layer_Wo(CudaModel* m, int layer) {
    return layer_param(m, layer, m->numHeads * 3);
}
static inline Param& layer_ff1W(CudaModel* m, int layer) {
    return layer_param(m, layer, m->numHeads * 3 + 1);
}
static inline Param& layer_ff1B(CudaModel* m, int layer) {
    return layer_param(m, layer, m->numHeads * 3 + 2);
}
static inline Param& layer_ff2W(CudaModel* m, int layer) {
    return layer_param(m, layer, m->numHeads * 3 + 3);
}
static inline Param& layer_ff2B(CudaModel* m, int layer) {
    return layer_param(m, layer, m->numHeads * 3 + 4);
}
static inline Param& layer_ln1G(CudaModel* m, int layer) {
    return layer_param(m, layer, m->numHeads * 3 + 5);
}
static inline Param& layer_ln1B(CudaModel* m, int layer) {
    return layer_param(m, layer, m->numHeads * 3 + 6);
}
static inline Param& layer_ln2G(CudaModel* m, int layer) {
    return layer_param(m, layer, m->numHeads * 3 + 7);
}
static inline Param& layer_ln2B(CudaModel* m, int layer) {
    return layer_param(m, layer, m->numHeads * 3 + 8);
}

// --- Forward pass for one transformer layer ---

static void layer_forward(CudaModel* m, int layer, float* x, int B, int T) {
    LayerBuffers* b = &m->layerBufs[layer];
    int BT = B * T;
    int d = m->headDim;
    int E = m->embedDim;
    float scale = 1.0f / sqrtf((float)d);

    // Pre-attention layer norm
    cuda_layernorm(x, layer_ln1G(m, layer).data, layer_ln1B(m, layer).data,
                    b->ln1Out, b->ln1Mean, b->ln1Rstd, BT, E);

    // Multi-head attention on normalized input
    for (int h = 0; h < m->numHeads; h++) {
        cuda_matmul(b->ln1Out, head_Wq(m, layer, h).data, b->headQ[h], BT, E, d);
        cuda_matmul(b->ln1Out, head_Wk(m, layer, h).data, b->headK[h], BT, E, d);
        cuda_matmul(b->ln1Out, head_Wv(m, layer, h).data, b->headV[h], BT, E, d);

        cuda_batched_attn_scores(b->headQ[h], b->headK[h], b->headScores[h], B, T, d, scale, 1);
        cuda_copy(b->headScores[h], b->headWeights[h], BT * T);
        cuda_softmax(b->headWeights[h], BT, T);

        cuda_batched_matmul(b->headWeights[h], b->headV[h], b->headOut[h],
                            B, T, T, d, (long long)T * T, (long long)T * d, (long long)T * d);
    }

    cuda_concat_heads((const float**)b->headOut, b->concat, BT, d, m->numHeads);
    cuda_matmul(b->concat, layer_Wo(m, layer).data, b->attnOut, BT, E, E);

    // Residual: afterAttn = x + attnOut
    cuda_add(x, b->attnOut, b->afterAttn, BT * E, E, BT * E);

    // Pre-FF layer norm
    cuda_layernorm(b->afterAttn, layer_ln2G(m, layer).data, layer_ln2B(m, layer).data,
                    b->ln2Out, b->ln2Mean, b->ln2Rstd, BT, E);

    // Feed-forward on normalized input
    cuda_matmul(b->ln2Out, layer_ff1W(m, layer).data, b->ff1, BT, E, m->ffDim);
    cuda_add(b->ff1, layer_ff1B(m, layer).data, b->ff1, BT * m->ffDim, m->ffDim, m->ffDim);
    cuda_tanh(b->ff1, b->ff1Act, BT * m->ffDim);

    cuda_matmul(b->ff1Act, layer_ff2W(m, layer).data, b->ff2, BT, m->ffDim, E);
    cuda_add(b->ff2, layer_ff2B(m, layer).data, b->ff2, BT * E, E, E);

    // Residual: afterFF = afterAttn + ff2
    cuda_add(b->afterAttn, b->ff2, b->afterFF, BT * E, E, BT * E);
}

// --- Backward pass for one transformer layer ---

static void layer_backward(CudaModel* m, int layer, float* x, float* dOut, int B, int T) {
    LayerBuffers* b = &m->layerBufs[layer];
    int BT = B * T;
    int d = m->headDim;
    int E = m->embedDim;
    int ffDim = m->ffDim;
    float scale = 1.0f / sqrtf((float)d);

    // --- FF backward ---
    // dOut flows into both the residual and the FF path
    // FF path: dOut → dLn2Out (through ln2 backward) → ff backward

    // ff2 backward: dLn2Out = dOut through FF
    cuda_zero(b->dFF1Act, BT * ffDim);
    cuda_transpose(layer_ff2W(m, layer).data, b->tmp, ffDim, E);
    cuda_matmul_add(dOut, b->tmp, b->dFF1Act, BT, E, ffDim);

    cuda_transpose(b->ff1Act, b->tmp, BT, ffDim);
    cuda_matmul_add(b->tmp, dOut, layer_ff2W(m, layer).grad, ffDim, BT, E);
    cuda_add_bias_backward(dOut, layer_ff2B(m, layer).grad, BT, E);

    cuda_zero(b->dFF1, BT * ffDim);
    cuda_tanh_backward(b->ff1Act, b->dFF1Act, b->dFF1, BT * ffDim);

    // dLn2Out from FF path
    cuda_zero(b->dLn2Out, BT * E);
    cuda_transpose(layer_ff1W(m, layer).data, b->tmp, E, ffDim);
    cuda_matmul_add(b->dFF1, b->tmp, b->dLn2Out, BT, ffDim, E);

    cuda_transpose(b->ln2Out, b->tmp, BT, E);
    cuda_matmul_add(b->tmp, b->dFF1, layer_ff1W(m, layer).grad, E, BT, ffDim);
    cuda_add_bias_backward(b->dFF1, layer_ff1B(m, layer).grad, BT, ffDim);

    // ln2 backward: dAfterAttn from ln2 (accumulates into dOut which is dAfterAttn)
    // dOut already has the residual gradient; add ln2 backward contribution
    cuda_layernorm_backward(b->dLn2Out, b->afterAttn,
                             layer_ln2G(m, layer).data, b->ln2Mean, b->ln2Rstd,
                             dOut, layer_ln2G(m, layer).grad, layer_ln2B(m, layer).grad,
                             BT, E);

    // --- Attention backward ---
    // dOut now contains dAfterAttn (residual + ln2 path)

    cuda_zero(b->dConcat, BT * E);
    cuda_transpose(layer_Wo(m, layer).data, b->tmp, E, E);
    cuda_matmul_add(dOut, b->tmp, b->dConcat, BT, E, E);

    cuda_transpose(b->concat, b->tmp, BT, E);
    cuda_matmul_add(b->tmp, dOut, layer_Wo(m, layer).grad, E, BT, E);

    cuda_split_heads(b->dConcat, b->dHeadOut, BT, d, m->numHeads);

    // dLn1Out accumulates attention gradients
    cuda_zero(b->dLn1Out, BT * E);

    for (int h = 0; h < m->numHeads; h++) {
        cuda_zero(b->dHeadWeights[h], BT * T);
        cuda_batched_attn_scores(b->dHeadOut[h], b->headV[h], b->dHeadWeights[h], B, T, d, 1.0f, 0);

        cuda_zero(b->dHeadV[h], BT * d);
        cuda_batched_transpose(b->headWeights[h], b->tmp, B, T, T);
        cuda_batched_matmul(b->tmp, b->dHeadOut[h], b->dHeadV[h],
                            B, T, T, d, (long long)T * T, (long long)T * d, (long long)T * d);

        cuda_batched_softmax_backward(b->headWeights[h], b->dHeadWeights[h], b->dHeadScores[h],
                                       B, T, scale, 1);

        cuda_zero(b->dHeadQ[h], BT * d);
        cuda_batched_matmul(b->dHeadScores[h], b->headK[h], b->dHeadQ[h],
                            B, T, T, d, (long long)T * T, (long long)T * d, (long long)T * d);

        cuda_zero(b->dHeadK[h], BT * d);
        cuda_batched_transpose(b->dHeadScores[h], b->tmp, B, T, T);
        cuda_batched_matmul(b->tmp, b->headQ[h], b->dHeadK[h],
                            B, T, T, d, (long long)T * T, (long long)T * d, (long long)T * d);

        // Weight gradients use ln1Out (normalized input to attention)
        cuda_transpose(b->ln1Out, b->tmp, BT, E);
        cuda_matmul_add(b->tmp, b->dHeadQ[h], head_Wq(m, layer, h).grad, E, BT, d);
        cuda_matmul_add(b->tmp, b->dHeadK[h], head_Wk(m, layer, h).grad, E, BT, d);
        cuda_matmul_add(b->tmp, b->dHeadV[h], head_Wv(m, layer, h).grad, E, BT, d);

        // Accumulate dLn1Out
        cuda_transpose(head_Wq(m, layer, h).data, b->tmp, E, d);
        cuda_matmul_add(b->dHeadQ[h], b->tmp, b->dLn1Out, BT, d, E);
        cuda_transpose(head_Wk(m, layer, h).data, b->tmp, E, d);
        cuda_matmul_add(b->dHeadK[h], b->tmp, b->dLn1Out, BT, d, E);
        cuda_transpose(head_Wv(m, layer, h).data, b->tmp, E, d);
        cuda_matmul_add(b->dHeadV[h], b->tmp, b->dLn1Out, BT, d, E);
    }

    // ln1 backward: propagate through pre-attention norm into dOut (= dx for previous layer)
    cuda_layernorm_backward(b->dLn1Out, x,
                             layer_ln1G(m, layer).data, b->ln1Mean, b->ln1Rstd,
                             dOut, layer_ln1G(m, layer).grad, layer_ln1B(m, layer).grad,
                             BT, E);
}

// --- Public API ---

extern "C" {

CudaModel* model_create(int vocabSize, int embedDim, int contextLen,
                          int numLayers, int numHeads) {
    CudaModel* m = new CudaModel();
    m->vocabSize = vocabSize;
    m->embedDim = embedDim;
    m->contextLen = contextLen;
    m->numLayers = numLayers;
    m->numHeads = numHeads;
    m->headDim = embedDim / numHeads;
    m->ffDim = embedDim * 4;

    m->paramsPerLayer = numHeads * 3 + 9;  // Wq,Wk,Wv per head + Wo,ff1W,ff1B,ff2W,ff2B,ln1G,ln1B,ln2G,ln2B
    m->numParams = 2 + numLayers * m->paramsPerLayer + 2;
    m->params = new Param[m->numParams];
    memset(m->params, 0, m->numParams * sizeof(Param));

    m->iTokenEmbed = 0;
    m->iPosEmbed = 1;
    m->iLayerBase = 2;
    m->iOutW = 2 + numLayers * m->paramsPerLayer;
    m->iOutB = m->iOutW + 1;

    m->layerBufs = nullptr;
    m->allocB = 0;
    m->allocT = 0;
    memset(&m->bufs, 0, sizeof(ModelBuffers));
    m->posCache = nullptr;
    m->posCacheSize = 0;
    m->posCacheT = 0;

    m->adam_m = nullptr;
    m->adam_v = nullptr;
    m->adamStep = 0;

    return m;
}

void model_set_param(CudaModel* m, int index, float* data, float* grad, int size) {
    m->params[index].data = data;
    m->params[index].grad = grad;
    m->params[index].size = size;
}

void model_alloc_buffers(CudaModel* m, int B, int T) {
    int BT = B * T;
    int oldBT = m->allocB * m->allocT;
    int oldT = m->allocT;
    bool needAlloc = BT > oldBT || T > oldT;  // grow if BT or T exceeds previous max

    if (needAlloc) {
        int maxT = T > oldT ? T : oldT;
        int maxBT = BT > oldBT ? BT : oldBT;
        int E = m->embedDim;
        int V = m->vocabSize;

        m->bufs.tokEmb = cuda_alloc(maxBT * E);
        m->bufs.posEmb = cuda_alloc(maxBT * E);
        m->bufs.x = cuda_alloc(maxBT * E);
        if (!m->bufs.layerInputs)
            m->bufs.layerInputs = (float**)malloc(m->numLayers * sizeof(float*));
        for (int i = 0; i < m->numLayers; i++)
            m->bufs.layerInputs[i] = cuda_alloc(maxBT * E);
        m->bufs.logits = cuda_alloc(maxBT * V);
        m->bufs.logProbs = cuda_alloc(maxBT * V);
        m->bufs.loss = cuda_alloc(1);
        m->bufs.dLogProbs = cuda_alloc(maxBT * V);
        m->bufs.dLogits = cuda_alloc(maxBT * V);
        m->bufs.dX = cuda_alloc(maxBT * E);
        int tmpSize = maxBT * E;
        if (maxBT * V > tmpSize) tmpSize = maxBT * V;
        if (E * V > tmpSize) tmpSize = E * V;
        m->bufs.tmp = cuda_alloc(tmpSize);
        m->bufs.tokens = cuda_alloc_int(maxBT);
        m->bufs.targets = cuda_alloc_int(maxBT);
        m->bufs.positions = cuda_alloc_int(maxBT);

        if (!m->layerBufs)
            m->layerBufs = new LayerBuffers[m->numLayers];
        for (int i = 0; i < m->numLayers; i++) {
            alloc_layer_bufs(&m->layerBufs[i], m->numHeads, maxBT, maxT,
                              m->headDim, E, m->ffDim);
        }

        m->allocB = B > m->allocB ? B : m->allocB;
        m->allocT = T > m->allocT ? T : m->allocT;
    }

    // Rebuild position cache when T changes (cheap CPU op)
    if (T != m->posCacheT || BT > m->posCacheSize) {
        if (!m->posCache || BT > m->posCacheSize) {
            if (m->posCache) free(m->posCache);
            int allocSize = BT > oldBT ? BT : oldBT;
            m->posCache = (int*)malloc(allocSize * sizeof(int));
            m->posCacheSize = allocSize;
        }
        for (int i = 0; i < BT; i++) m->posCache[i] = i % T;
        m->posCacheT = T;
    }
}

float model_forward(CudaModel* m, const int* tokens, const int* targets, int B, int T) {
    model_alloc_buffers(m, B, T);
    int BT = B * T;
    int E = m->embedDim;
    int V = m->vocabSize;
    ModelBuffers* b = &m->bufs;

    cuda_copy_to_gpu_int(b->tokens, tokens, BT);
    cuda_copy_to_gpu_int(b->targets, targets, BT);
    cuda_copy_to_gpu_int(b->positions, m->posCache, BT);

    cuda_gather(m->params[m->iTokenEmbed].data, b->tokens, b->tokEmb, BT, E);
    cuda_gather(m->params[m->iPosEmbed].data, b->positions, b->posEmb, BT, E);
    cuda_add(b->tokEmb, b->posEmb, b->x, BT * E, E, BT * E);

    float* current = b->x;
    for (int i = 0; i < m->numLayers; i++) {
        cuda_copy(current, b->layerInputs[i], BT * E);
        layer_forward(m, i, current, B, T);
        current = m->layerBufs[i].afterFF;
    }

    cuda_matmul(current, m->params[m->iOutW].data, b->logits, BT, E, V);
    cuda_add(b->logits, m->params[m->iOutB].data, b->logits, BT * V, V, V);
    cuda_log_softmax(b->logits, b->logProbs, BT, V);

    cuda_nll_loss(b->logProbs, b->targets, b->loss, BT, V);
    cudaDeviceSynchronize();

    float loss;
    cuda_copy_to_cpu(&loss, b->loss, 1);
    return loss;
}

// Forward without targets (for generate)
void model_forward_logprobs(CudaModel* m, const int* tokens, int B, int T) {
    model_alloc_buffers(m, B, T);
    int BT = B * T;
    int E = m->embedDim;
    int V = m->vocabSize;
    ModelBuffers* b = &m->bufs;

    cuda_copy_to_gpu_int(b->tokens, tokens, BT);
    cuda_copy_to_gpu_int(b->positions, m->posCache, BT);

    cuda_gather(m->params[m->iTokenEmbed].data, b->tokens, b->tokEmb, BT, E);
    cuda_gather(m->params[m->iPosEmbed].data, b->positions, b->posEmb, BT, E);
    cuda_add(b->tokEmb, b->posEmb, b->x, BT * E, E, BT * E);

    float* current = b->x;
    for (int i = 0; i < m->numLayers; i++) {
        cuda_copy(current, b->layerInputs[i], BT * E);
        layer_forward(m, i, current, B, T);
        current = m->layerBufs[i].afterFF;
    }

    cuda_matmul(current, m->params[m->iOutW].data, b->logits, BT, E, V);
    cuda_add(b->logits, m->params[m->iOutB].data, b->logits, BT * V, V, V);
    cuda_log_softmax(b->logits, b->logProbs, BT, V);
    cudaDeviceSynchronize();
}

void model_get_logprobs(CudaModel* m, float* out, int size) {
    cuda_copy_to_cpu(out, m->bufs.logProbs, size);
}

void model_backward(CudaModel* m, int B, int T) {
    int BT = B * T;
    int E = m->embedDim;
    int V = m->vocabSize;
    ModelBuffers* b = &m->bufs;

    cuda_zero(b->dLogProbs, BT * V);
    cuda_nll_loss_backward(b->dLogProbs, b->targets, BT, V, 1.0f);

    cuda_zero(b->dLogits, BT * V);
    cuda_log_softmax_backward(b->logProbs, b->dLogProbs, b->dLogits, BT, V);

    float* lastOut = m->layerBufs[m->numLayers - 1].afterFF;
    cuda_transpose(lastOut, b->tmp, BT, E);
    cuda_matmul_add(b->tmp, b->dLogits, m->params[m->iOutW].grad, E, BT, V);
    cuda_add_bias_backward(b->dLogits, m->params[m->iOutB].grad, BT, V);

    cuda_zero(b->dX, BT * E);
    cuda_transpose(m->params[m->iOutW].data, b->tmp, E, V);
    cuda_matmul_add(b->dLogits, b->tmp, b->dX, BT, V, E);

    float* dCurrent = b->dX;
    for (int i = m->numLayers - 1; i >= 0; i--) {
        layer_backward(m, i, b->layerInputs[i], dCurrent, B, T);
    }

    cuda_gather_backward(m->params[m->iTokenEmbed].grad, b->tokens, dCurrent, BT, E);
    cuda_gather_backward(m->params[m->iPosEmbed].grad, b->positions, dCurrent, BT, E);
}

void model_zero_grad(CudaModel* m) {
    for (int i = 0; i < m->numParams; i++) {
        if (m->params[i].grad)
            cuda_zero(m->params[i].grad, m->params[i].size);
    }
}

void model_update(CudaModel* m, float lr) {
    // Lazy-init AdamW moment buffers (zeroed by cudaMalloc + cuda_zero)
    if (!m->adam_m) {
        m->adam_m = (float**)malloc(m->numParams * sizeof(float*));
        m->adam_v = (float**)malloc(m->numParams * sizeof(float*));
        for (int i = 0; i < m->numParams; i++) {
            m->adam_m[i] = cuda_alloc(m->params[i].size);
            m->adam_v[i] = cuda_alloc(m->params[i].size);
            cuda_zero(m->adam_m[i], m->params[i].size);
            cuda_zero(m->adam_v[i], m->params[i].size);
        }
    }

    m->adamStep++;
    for (int i = 0; i < m->numParams; i++) {
        cuda_adamw(m->params[i].data, m->params[i].grad,
                    m->adam_m[i], m->adam_v[i],
                    lr, 0.9f, 0.999f, 1e-8f, 0.01f,
                    m->adamStep, m->params[i].size);
    }
    cudaDeviceSynchronize();
}

// Single training step: batch prep + zero grad + forward + backward + update.
// all_tokens: full token array on CPU, tokenCount: length of all_tokens
// offsets: array of B random start positions (CPU), lr: learning rate
// Returns loss value.
float model_train_step(CudaModel* m, const int* all_tokens, int tokenCount,
                        const int* offsets, int B, int T, float lr) {
    // Build batch on CPU
    int BT = B * T;
    int* batchInput = (int*)malloc(BT * sizeof(int));
    int* batchTarget = (int*)malloc(BT * sizeof(int));
    for (int b = 0; b < B; b++) {
        int start = offsets[b];
        for (int t = 0; t < T; t++) {
            batchInput[b * T + t] = all_tokens[start + t];
            batchTarget[b * T + t] = all_tokens[start + 1 + t];
        }
    }

    // Zero gradients
    for (int i = 0; i < m->numParams; i++) {
        if (m->params[i].grad)
            cuda_zero(m->params[i].grad, m->params[i].size);
    }

    // Forward
    model_alloc_buffers(m, B, T);
    int E = m->embedDim;
    int V = m->vocabSize;
    ModelBuffers* buf = &m->bufs;

    cuda_copy_to_gpu_int(buf->tokens, batchInput, BT);
    cuda_copy_to_gpu_int(buf->targets, batchTarget, BT);
    cuda_copy_to_gpu_int(buf->positions, m->posCache, BT);

    cuda_gather(m->params[m->iTokenEmbed].data, buf->tokens, buf->tokEmb, BT, E);
    cuda_gather(m->params[m->iPosEmbed].data, buf->positions, buf->posEmb, BT, E);
    cuda_add(buf->tokEmb, buf->posEmb, buf->x, BT * E, E, BT * E);

    float* current = buf->x;
    for (int i = 0; i < m->numLayers; i++) {
        cuda_copy(current, buf->layerInputs[i], BT * E);
        layer_forward(m, i, current, B, T);
        current = m->layerBufs[i].afterFF;
    }

    cuda_matmul(current, m->params[m->iOutW].data, buf->logits, BT, E, V);
    cuda_add(buf->logits, m->params[m->iOutB].data, buf->logits, BT * V, V, V);
    cuda_log_softmax(buf->logits, buf->logProbs, BT, V);
    cuda_nll_loss(buf->logProbs, buf->targets, buf->loss, BT, V);
    cudaDeviceSynchronize();

    float loss;
    cuda_copy_to_cpu(&loss, buf->loss, 1);

    // Backward
    model_backward(m, B, T);

    // Update
    model_update(m, lr);

    free(batchInput);
    free(batchTarget);
    return loss;
}

// Generate tokens. Returns number of tokens written to out_tokens (includes seed).
// out_tokens must be large enough for seedLen + numTokens.
// callback: if non-null, called with each new token index. Return 0 to stop early.
int model_generate(CudaModel* m, const int* seedTokens, int seedLen,
                    int numTokens, float temperature, int* out_tokens,
                    int (*callback)(int token, void* user), void* user) {
    int V = m->vocabSize;
    int maxCtx = m->contextLen;

    // Copy seed to output
    int totalLen = seedLen;
    memcpy(out_tokens, seedTokens, seedLen * sizeof(int));

    // CPU buffer for logprobs of last position
    float* probs = (float*)malloc(V * sizeof(float));

    // Simple xorshift64 RNG seeded from time
    unsigned long long rng_state = (unsigned long long)time(nullptr) ^ 0xdeadbeef;

    for (int i = 0; i < numTokens; i++) {
        // Use last maxCtx tokens as context
        int ctxStart = totalLen > maxCtx ? totalLen - maxCtx : 0;
        int T = totalLen - ctxStart;

        // Forward pass (B=1)
        model_forward_logprobs(m, out_tokens + ctxStart, 1, T);

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[gen] CUDA error: %s\n", cudaGetErrorString(err));
            free(probs);
            return totalLen;
        }

        // Copy last position's logprobs to CPU — only the last row
        float* lastLogProbs = (float*)malloc(V * sizeof(float));
        cuda_copy_to_cpu(lastLogProbs, m->bufs.logProbs + (T - 1) * V, V);

        // Temperature-scaled softmax sampling from last position
        float maxVal = -1e30f;
        for (int j = 0; j < V; j++) {
            float v = lastLogProbs[j] / temperature;
            if (v > maxVal) maxVal = v;
            probs[j] = v;
        }
        float sumExp = 0.0f;
        for (int j = 0; j < V; j++) {
            probs[j] = expf(probs[j] - maxVal);
            sumExp += probs[j];
        }
        for (int j = 0; j < V; j++) probs[j] /= sumExp;

        // Sample using xorshift64
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        float r = (float)(rng_state & 0xFFFFFF) / (float)0xFFFFFF;

        float cumsum = 0.0f;
        int next = 0;
        for (int j = 0; j < V; j++) {
            cumsum += probs[j];
            if (r < cumsum) { next = j; break; }
        }

        free(lastLogProbs);

        out_tokens[totalLen] = next;
        totalLen++;

        if (callback && !callback(next, user)) break;
    }

    free(probs);
    return totalLen;
}

void model_free(CudaModel* m) {
    delete[] m->params;
    if (m->posCache) free(m->posCache);
    // Buffer cleanup omitted for brevity (process exit cleans GPU)
    delete m;
}

} // extern "C"
