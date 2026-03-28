// Interactive text generator using CUDA-accelerated model with BPE
// Run: node generate.js
// Run: node generate.js --temp 0.7

const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { GpuMiniLLM } = require("./gpu-model");
const { cuda } = require("./gpu-tensor");

const args = process.argv.slice(2);
const getArg = (name, def) => {
  const i = args.indexOf("--" + name);
  return i >= 0 ? args[i + 1] : def;
};
const defaultTemp = parseFloat(getArg("temp", "0.7"));
const vocabPath = path.resolve(getArg("vocab", path.join(__dirname, "vocab.json")));

// Load weights
const weightsPath = path.join(__dirname, "weights.bin");
if (!fs.existsSync(weightsPath)) {
  console.log("No weights.bin found. Train first:\n  node train.js\n");
  process.exit(1);
}

// Load BPE vocab
if (!fs.existsSync(vocabPath)) {
  console.error("No vocab.json found. Train vocabulary first:\n  node train-vocab.js\n");
  process.exit(1);
}

console.log("Loading trained model...");
const buf = fs.readFileSync(weightsPath);
const header = new Int32Array(buf.buffer, buf.byteOffset, 5);
const [vocabSize, embedDim, contextLen, numLayers, numHeads] = header;

const bpe = cuda.bpeCreate();
cuda.bpeLoad(bpe, vocabPath);

// Verify vocab matches model
const bpeVocab = cuda.bpeVocabSize(bpe);
if (bpeVocab !== vocabSize) {
  console.error(`Vocab mismatch: vocab.json has ${bpeVocab} tokens, model expects ${vocabSize}`);
  process.exit(1);
}

const model = new GpuMiniLLM(vocabSize, embedDim, contextLen, numLayers, numHeads);
model.uploadWeightsBin(weightsPath);

console.log(`Loaded: ${numLayers} layers, ${numHeads} heads, ${model.paramCount().toLocaleString()} params (CUDA)`);
console.log(`Tokenizer: BPE (vocab ${vocabSize})\n`);
console.log("Type a prompt and press Enter to generate text.");
console.log("Commands: /temp 0.5, /len 200, /quit\n");

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
let temp = defaultTemp;
let genLen = 200;

function prompt() {
  rl.question("> ", (input) => {
    if (!input || input === "/quit") {
      cuda.bpeFree(bpe);
      rl.close();
      return;
    }

    if (input.startsWith("/temp ")) {
      temp = parseFloat(input.slice(6));
      console.log(`Temperature set to ${temp}\n`);
      return prompt();
    }

    if (input.startsWith("/len ")) {
      genLen = parseInt(input.slice(5));
      console.log(`Generation length set to ${genLen}\n`);
      return prompt();
    }

    const processed = input.replace(/\\n/g, "\n");
    const inputTokens = Array.from(cuda.bpeEncode(bpe, processed));

    const generated = model.generate(inputTokens, genLen, temp);
    const fullText = cuda.bpeDecode(bpe, new Int32Array(generated));
    const continuation = fullText.slice(processed.length);
    console.log("\n" + processed + continuation + "\n");
    prompt();
  });
}

prompt();
