#!/bin/bash
# Build the CUDA addon for Node.js
# Prerequisites: CUDA toolkit installed, nvcc in PATH, node-gyp available
#
# Run: bash build.sh

set -e

ARCH="${1:-sm_89}"

echo "=== Building CUDA Mini LLM addon ==="
echo "GPU architecture: $ARCH"

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Install CUDA toolkit first:"
    echo "  https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"
echo "Node version: $(node --version)"

# Step 1: Compile CUDA kernels
echo ""
echo "Compiling CUDA kernels..."
mkdir -p build/Release/obj.target
nvcc -c native/kernels.cu \
    -o build/Release/obj.target/kernels.o \
    --compiler-options '-fPIC' \
    -O3 \
    -arch=$ARCH

nvcc -c native/model.cu \
    -o build/Release/obj.target/model.o \
    --compiler-options '-fPIC' \
    -O3 \
    -arch=$ARCH

g++ -c -O2 -std=c++17 -fPIC native/bpe.cpp \
    -o build/Release/obj.target/bpe.o

echo "CUDA + BPE compiled."

# Step 2: Build Node.js addon (links to the compiled CUDA kernels)
echo ""
echo "Building Node.js addon..."
npx node-gyp configure build

echo ""
echo "Build complete! Test with:"
echo "  node -e \"const c = require('./build/Release/cuda_addon.node'); console.log('CUDA addon loaded:', Object.keys(c).length, 'functions')\""
