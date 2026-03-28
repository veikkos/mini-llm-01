// Pre-encode text into a binary token file for fast training
// Run: node encode.js --input data/tinystories.txt
// Run: node encode.js --input data/tinystories.txt --output tokens.bin

const fs = require("fs");
const path = require("path");
const { cuda } = require("./gpu-tensor");

const args = process.argv.slice(2);
const getArg = (name, def) => {
  const i = args.indexOf("--" + name);
  return i >= 0 ? args[i + 1] : def;
};

const inputPath = path.resolve(getArg("input", path.join(__dirname, "data/tinystories.txt")));
const vocabPath = path.resolve(getArg("vocab", path.join(__dirname, "vocab.json")));
const outputPath = path.resolve(getArg("output", path.join(__dirname, "tokens.bin")));

console.log("=== BPE Encode ===\n");

if (!fs.existsSync(vocabPath)) {
  console.error("No vocab.json found. Train vocabulary first:\n  node train-vocab.js\n");
  process.exit(1);
}

const bpe = cuda.bpeCreate();
cuda.bpeLoad(bpe, vocabPath);
const vocabSize = cuda.bpeVocabSize(bpe);
console.log(`Vocab: ${vocabSize} tokens`);

const maxChars = parseInt(getArg("maxchars", "0"));
process.stdout.write(`Loading ${path.basename(inputPath)}...`);
let text = fs.readFileSync(inputPath, "utf-8");
console.log(` ${(text.length / 1e6).toFixed(1)}MB`);
if (maxChars > 0 && text.length > maxChars) {
  text = text.slice(0, maxChars);
  console.log(`Truncated to ${(maxChars / 1e6).toFixed(0)}MB`);
}

// Encode in chunks so we can show progress + ETA
const CHUNK_SIZE = 1_000_000; // 1MB per chunk
const numChunks = Math.ceil(text.length / CHUNK_SIZE);
const allTokens = [];
const start = Date.now();

console.log(`Encoding ${(text.length / 1e6).toFixed(1)}MB in ${numChunks} chunks...`);
for (let i = 0; i < numChunks; i++) {
  const chunk = text.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
  const chunkTokens = cuda.bpeEncode(bpe, chunk);
  allTokens.push(chunkTokens);

  const done = i + 1;
  const elapsedS = (Date.now() - start) / 1000;
  const etaS = (elapsedS / done) * (numChunks - done);
  process.stdout.write(`\r  Chunk ${done}/${numChunks} | ${elapsedS.toFixed(0)}s elapsed | ~${etaS.toFixed(0)}s remaining`);
}
console.log();

// Merge chunks into single Int32Array
const totalLen = allTokens.reduce((s, t) => s + t.length, 0);
const tokens = new Int32Array(totalLen);
let offset = 0;
for (const t of allTokens) {
  tokens.set(t, offset);
  offset += t.length;
}

const elapsed = ((Date.now() - start) / 1000).toFixed(1);
console.log(`Encoded ${tokens.length.toLocaleString()} tokens in ${elapsed}s`);
console.log(`Compression: ${(text.length / tokens.length).toFixed(1)} chars/token`);

// Save as raw Int32Array binary
const buf = Buffer.from(tokens.buffer, tokens.byteOffset, tokens.byteLength);
fs.writeFileSync(outputPath, buf);
const sizeMB = (buf.length / 1e6).toFixed(1);
console.log(`\nSaved: ${outputPath} (${sizeMB}MB)`);
console.log(`\nTrain with:\n  node train.js --tokens ${path.basename(outputPath)}`);

cuda.bpeFree(bpe);
