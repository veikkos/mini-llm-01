@echo off
REM Build the CUDA addon for Node.js on Windows
REM Run from Developer Command Prompt: build.cmd

setlocal

echo === Building CUDA Mini LLM addon ===

where nvcc >nul 2>&1
if errorlevel 1 (
    echo Error: nvcc not found. Install CUDA toolkit and add to PATH.
    exit /b 1
)

nvcc --version | findstr release
node --version

echo.
echo Compiling CUDA kernels...
if not exist build\Release\obj.target mkdir build\Release\obj.target

nvcc -c native/kernels.cu -o build/Release/obj.target/kernels.obj -O3 -arch=sm_89 -lcublas --no-compress 2>&1
if errorlevel 1 (
    echo CUDA kernels compilation failed.
    exit /b 1
)

nvcc -c native/model.cu -o build/Release/obj.target/model.obj -O3 -arch=sm_89 -lcublas --no-compress 2>&1
if errorlevel 1 (
    echo CUDA model compilation failed.
    exit /b 1
)
echo Compiling BPE tokenizer...
cl /c /O2 /std:c++17 /EHsc native/bpe.cpp /Fo:build/Release/obj.target/bpe.obj 2>&1
if errorlevel 1 (
    echo BPE compilation failed.
    exit /b 1
)
echo CUDA + BPE compiled.

echo.
echo Building Node.js addon...
call npx node-gyp configure build
if errorlevel 1 (
    echo node-gyp build failed.
    exit /b 1
)

echo.
echo Build complete! Test with:
echo   node -e "const c = require('./build/Release/cuda_addon.node'); console.log('CUDA addon loaded:', Object.keys(c).length, 'functions')"
