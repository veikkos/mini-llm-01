// Train the CUDA-accelerated Mini LLM with BPE tokenizer
// Run: node train.js --tokens tokens.bin
// Run: node train.js --tokens tokens.bin --steps 50000 --embed 256 --layers 6 --heads 8

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

// Load pre-encoded tokens
const tokensPath = getArg("tokens", null);
if (!tokensPath) {
  console.error("Missing --tokens argument. Encode first:\n  node encode.js --input data/tinystories.txt\n");
  process.exit(1);
}
console.log("=== Mini LLM — CUDA Training ===\n");
console.log(`Backend: CUDA`);
console.log(`Tokenizer: BPE (vocab ${vocabSize})`);

const p = path.resolve(tokensPath);
process.stdout.write(`Loading pre-encoded tokens from ${path.basename(p)}...`);
const buf = fs.readFileSync(p);
const tokens = new Int32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
console.log(` ${tokens.length.toLocaleString()} tokens`);

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

// Save weights on Ctrl+C instead of losing progress
let interrupted = false;
process.on("SIGINT", () => {
  if (interrupted) process.exit(1); // second Ctrl+C force-quits
  console.log("\n\nInterrupted — saving weights...");
  interrupted = true;
});

const offsets = new Int32Array(batchSize);
for (let step = 0; step < numSteps && !interrupted; step++) {
  for (let b = 0; b < batchSize; b++) {
    offsets[b] = Math.floor(Math.random() * (tokens.length - contextLen - 1));
  }

  const lr = getLR(step);
  const loss = model.trainStep(tokens, offsets, batchSize, contextLen, lr);

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
