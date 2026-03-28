// GPU Tensor — all data lives on the GPU.
//
// Unlike 05's tensor.js where we build a computation graph and call backward(),
// here we manage forward and backward passes explicitly. This avoids the overhead
// of graph construction and lets us keep tensors on GPU between operations.
//
// Each GpuTensor is just a pointer to GPU memory + shape metadata.

const cuda = require("./build/Release/cuda_addon.node");

class GpuTensor {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.size = rows * cols;
    this.data = cuda.alloc(this.size);
    this.grad = cuda.alloc(this.size);
    cuda.zero(this.data, this.size);
    cuda.zero(this.grad, this.size);
  }

  // Upload CPU Float32Array to GPU
  upload(cpuData) {
    cuda.toGPU(this.data, cpuData);
  }

  // Download GPU data to CPU
  download() {
    const buf = new Float32Array(this.size);
    cuda.toCPU(buf, this.data, this.size);
    return buf;
  }

  downloadGrad() {
    const buf = new Float32Array(this.size);
    cuda.toCPU(buf, this.grad, this.size);
    return buf;
  }

  zeroGrad() {
    cuda.zero(this.grad, this.size);
  }

  free() {
    cuda.free(this.data);
    cuda.free(this.grad);
  }
}

// Scratch/intermediate GPU buffers (not learnable parameters)
class GpuBuffer {
  constructor(size) {
    this.ptr = cuda.alloc(size);
    this.size = size;
  }

  free() {
    cuda.free(this.ptr);
  }
}

// Integer buffer on GPU (for token indices)
class GpuIntBuffer {
  constructor(size) {
    this.ptr = cuda.allocInt(size);
    this.size = size;
  }

  upload(int32Array) {
    cuda.toGPUInt(this.ptr, int32Array);
  }

  free() {
    cuda.free(this.ptr);
  }
}

// Helper: create GpuTensor with random Xavier init
function gpuRandn(rows, cols, extraScale = 1) {
  const t = new GpuTensor(rows, cols);
  const scale = Math.sqrt(2 / (rows + cols)) * extraScale;
  const data = new Float32Array(rows * cols);
  for (let i = 0; i < data.length; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
  }
  t.upload(data);
  return t;
}

function gpuZeros(rows, cols) {
  return new GpuTensor(rows, cols);
}

function gpuOnes(rows, cols) {
  const t = new GpuTensor(rows, cols);
  const data = new Float32Array(rows * cols).fill(1.0);
  t.upload(data);
  return t;
}

module.exports = { GpuTensor, GpuBuffer, GpuIntBuffer, gpuRandn, gpuZeros, gpuOnes, cuda };
