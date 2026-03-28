// Train the CUDA-accelerated Mini LLM with BPE tokenizer
// Run: node train.js
// Run: node train.js --steps 50000 --embed 256 --layers 6 --heads 8
// Run: node train.js --tokens tokens.bin   (pre-encoded, skips BPE encoding)

const fs = require("fs");
const path = require("path");
const { GpuMiniLLM } = require("./gpu-model");
const { cuda } = require("./gpu-tensor");

// Parse args
const args = process.argv.slice(2);
const getArg = (name, def) => {
  const i = args.indexOf("--" + name);
  return i >= 0 ? args[i + 1] : def;
};
const getIntArg = (name, def) => parseInt(getArg(name, String(def)));

const numSteps = getIntArg("steps", 50000);
const embedDim = getIntArg("embed", 256);
const numLayers = getIntArg("layers", 6);
const numHeads = getIntArg("heads", 8);
const contextLen = getIntArg("context", 128);
const batchSize = getIntArg("batch", 32);
const noGen = args.includes("--no-gen");
const vocabPath = path.resolve(getArg("vocab", path.join(__dirname, "vocab.json")));

// Load BPE vocab
if (!fs.existsSync(vocabPath)) {
  console.error("No vocab.json found. Train vocabulary first:\n  node train-vocab.js\n");
  process.exit(1);
}

const bpe = cuda.bpeCreate();
cuda.bpeLoad(bpe, vocabPath);
const vocabSize = cuda.bpeVocabSize(bpe);
const eosToken = cuda.bpeEosToken();

// Load tokens — either from pre-encoded binary or by encoding text on the fly
const tokensPath = getArg("tokens", null);
console.log("=== Mini LLM — CUDA Training ===\n");
console.log(`Backend: CUDA`);
console.log(`Tokenizer: BPE (vocab ${vocabSize})`);

let tokens;
if (tokensPath) {
  const p = path.resolve(tokensPath);
  process.stdout.write(`Loading pre-encoded tokens from ${path.basename(p)}...`);
  const buf = fs.readFileSync(p);
  tokens = new Int32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  console.log(` ${tokens.length.toLocaleString()} tokens`);
} else {
  const inputPath = path.resolve(getArg("input", path.join(__dirname, "data/tinystories.txt")));
  process.stdout.write(`Loading ${path.basename(inputPath)}...`);
  const text = fs.readFileSync(inputPath, "utf-8");
  console.log(` ${(text.length / 1e6).toFixed(1)}MB`);

  process.stdout.write("Encoding text with BPE...");
  const encStart = Date.now();
  tokens = Array.from(cuda.bpeEncode(bpe, text));
  console.log(` ${tokens.length.toLocaleString()} tokens (${((Date.now() - encStart) / 1000).toFixed(1)}s)`);
}

// Create model
const model = new GpuMiniLLM(vocabSize, embedDim, contextLen, numLayers, numHeads);
console.log(`Model: ${numLayers} layers, ${numHeads} heads, embedDim=${embedDim}, contextLen=${contextLen}`);
console.log(`Parameters: ${model.paramCount().toLocaleString()}`);
console.log(`Training for ${numSteps} steps, batch ${batchSize}...\n`);

// Train — AdamW with warmup + cosine LR schedule
const maxLR = 0.001;
const minLR = 1e-5;
const warmupSteps = 500;
const startTime = Date.now();
const seedText = getArg("seed", "Once upon a time");
const seed = Array.from(cuda.bpeEncode(bpe, seedText));

function getLR(step) {
  if (step < warmupSteps) return maxLR * (step + 1) / warmupSteps;
  const decay = (step - warmupSteps) / (numSteps - warmupSteps);
  return minLR + 0.5 * (maxLR - minLR) * (1 + Math.cos(Math.PI * decay));
}

for (let step = 0; step < numSteps; step++) {
  const batchInput = new Int32Array(batchSize * contextLen);
  const batchTarget = new Int32Array(batchSize * contextLen);
  for (let b = 0; b < batchSize; b++) {
    const start = Math.floor(Math.random() * (tokens.length - contextLen - 1));
    for (let t = 0; t < contextLen; t++) {
      batchInput[b * contextLen + t] = tokens[start + t];
      batchTarget[b * contextLen + t] = tokens[start + 1 + t];
    }
  }

  const lr = getLR(step);
  model.zeroGrad();
  const loss = model.forward(batchInput, batchTarget, batchSize);
  model.backward(batchSize, contextLen);
  model.update(lr);

  if (step % 500 === 0 || step === numSteps - 1) {
    const elapsedS = (Date.now() - startTime) / 1000;
    const stepsPerSec = step > 0 ? (step / elapsedS).toFixed(1) : "0.0";
    const etaS = step > 0 ? Math.round((numSteps - step) / (step / elapsedS)) : 0;
    const etaMin = Math.floor(etaS / 60);
    const etaSec = etaS % 60;
    const eta = etaMin > 0 ? `${etaMin}m${String(etaSec).padStart(2, "0")}s` : `${etaSec}s`;
    console.log(`  Step ${String(step).padStart(5)} | Loss: ${loss.toFixed(3)} | lr: ${lr.toFixed(6)} | ${elapsedS.toFixed(0)}s (${stepsPerSec} step/s) | ~${eta} remaining`);
    if (!noGen) {
      const genTokens = model.generate(seed, 80, 0.8);
      const sample = cuda.bpeDecode(bpe, new Int32Array(genTokens)).slice(0, 120);
      console.log(`         | "${sample}..."`);
    }
    console.log();
  }
}

const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
console.log(`Training complete in ${totalTime}s\n`);

// Save weights
const weightsPath = path.join(__dirname, "weights.bin");
model.saveWeightsBin(weightsPath);
const sizeMB = (fs.statSync(weightsPath).size / 1024 / 1024).toFixed(1);
console.log(`Weights saved to weights.bin (${sizeMB}MB)`);
console.log("Run: node generate.js");

cuda.bpeFree(bpe);
