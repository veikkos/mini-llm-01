// GPU-accelerated Mini Transformer Language Model
//
// Weight initialization and management in JS.
// Forward/backward/update orchestration in C++ (native/model.cu)
// for minimal JS→CUDA round trips.

const { GpuTensor, gpuRandn, gpuZeros, gpuOnes, cuda } = require("./gpu-tensor");

class GpuMiniLLM {
  constructor(vocabSize, embedDim = 256, contextLen = 128, numLayers = 6, numHeads = 8) {
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.contextLen = contextLen;
    this.numLayers = numLayers;
    this.numHeads = numHeads;

    const headDim = Math.floor(embedDim / numHeads);
    const resScale = 1 / Math.sqrt(2 * numLayers);

    // Embeddings
    this.tokenEmbed = gpuRandn(vocabSize, embedDim);
    this.posEmbed = gpuRandn(contextLen, embedDim);

    // Transformer layers
    this.layerParams = [];
    for (let i = 0; i < numLayers; i++) {
      const layer = {
        heads: [],
        Wo: gpuRandn(headDim * numHeads, embedDim, resScale),
        ff1W: gpuRandn(embedDim, embedDim * 4),
        ff1B: gpuZeros(1, embedDim * 4),
        ff2W: gpuRandn(embedDim * 4, embedDim, resScale),
        ff2B: gpuZeros(1, embedDim),
        ln1G: gpuOnes(1, embedDim),   // layer norm gamma (init 1)
        ln1B: gpuZeros(1, embedDim),  // layer norm beta (init 0)
        ln2G: gpuOnes(1, embedDim),
        ln2B: gpuZeros(1, embedDim),
      };
      for (let h = 0; h < numHeads; h++) {
        layer.heads.push({
          Wq: gpuRandn(embedDim, headDim),
          Wk: gpuRandn(embedDim, headDim),
          Wv: gpuRandn(embedDim, headDim),
        });
      }
      this.layerParams.push(layer);
    }

    // Output
    this.outW = gpuRandn(embedDim, vocabSize);
    this.outB = gpuZeros(1, vocabSize);

    // Collect all params in the order model.cu expects:
    // tokenEmbed, posEmbed, [layer: heads×(Wq,Wk,Wv), Wo, ff1W, ff1B, ff2W, ff2B, ln1G, ln1B, ln2G, ln2B], outW, outB
    this.params = [this.tokenEmbed, this.posEmbed];
    for (const layer of this.layerParams) {
      for (const head of layer.heads) {
        this.params.push(head.Wq, head.Wk, head.Wv);
      }
      this.params.push(layer.Wo, layer.ff1W, layer.ff1B, layer.ff2W, layer.ff2B,
                        layer.ln1G, layer.ln1B, layer.ln2G, layer.ln2B);
    }
    this.params.push(this.outW, this.outB);

    // Create C++ model handle and register params
    this._handle = cuda.modelCreate(vocabSize, embedDim, contextLen, numLayers, numHeads);
    for (let i = 0; i < this.params.length; i++) {
      cuda.modelSetParam(this._handle, i, this.params[i].data, this.params[i].grad, this.params[i].size);
    }
  }

  paramCount() {
    return this.params.reduce((s, p) => s + p.size, 0);
  }

  forward(tokenArr, targetArr, B = 1) {
    const tokens = tokenArr instanceof Int32Array ? tokenArr : new Int32Array(tokenArr);
    const targets = targetArr instanceof Int32Array ? targetArr : new Int32Array(targetArr);
    const T = tokens.length / B;
    return cuda.modelForward(this._handle, tokens, targets, B, T);
  }

  backward(B, T) {
    cuda.modelBackward(this._handle, B, T);
  }

  zeroGrad() {
    cuda.modelZeroGrad(this._handle);
  }

  update(lr) {
    cuda.modelUpdate(this._handle, lr);
  }

  // Generate tokens — entire loop runs in C++
  generate(startTokens, numTokens, temperature = 0.8) {
    const seed = new Int32Array(startTokens);
    return Array.from(cuda.modelGenerate(this._handle, seed, numTokens, temperature));
  }

  // Streaming generate — calls callback(tokenIndex) per token, return false to stop
  generateStream(startTokens, numTokens, temperature, callback) {
    const seed = new Int32Array(startTokens);
    return Array.from(cuda.modelGenerateStream(this._handle, seed, numTokens, temperature, callback));
  }

  downloadWeights() {
    return this.params.map(p => ({
      shape: [p.rows, p.cols],
      data: Array.from(p.download()),
    }));
  }

  uploadWeights(paramData) {
    for (let i = 0; i < this.params.length; i++) {
      this.params[i].upload(new Float32Array(paramData[i].data));
    }
    // Re-register pointers with C++ model (same pointers, but just in case)
    for (let i = 0; i < this.params.length; i++) {
      cuda.modelSetParam(this._handle, i, this.params[i].data, this.params[i].grad, this.params[i].size);
    }
  }
}

module.exports = { GpuMiniLLM };
