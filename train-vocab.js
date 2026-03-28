// Train a BPE vocabulary from a text corpus
// Run: node train-vocab.js
// Run: node train-vocab.js --input data/tinystories.txt --output vocab.json --size 4096

const fs = require("fs");
const path = require("path");
const { cuda } = require("./gpu-tensor");

const args = process.argv.slice(2);
const getArg = (name, def) => {
  const i = args.indexOf("--" + name);
  return i >= 0 ? args[i + 1] : def;
};

const inputPath = path.resolve(getArg("input", path.join(__dirname, "data/tinystories.txt")));
const outputPath = path.resolve(getArg("output", path.join(__dirname, "vocab.json")));
const vocabSize = parseInt(getArg("size", "4096"));

console.log("=== BPE Vocabulary Training ===\n");
console.log(`Input:      ${inputPath}`);
console.log(`Output:     ${outputPath}`);
console.log(`Vocab size: ${vocabSize}`);

if (!fs.existsSync(inputPath)) {
  console.error(`\nError: input file not found: ${inputPath}`);
  process.exit(1);
}

const maxChars = parseInt(getArg("maxchars", "0"));
let text = fs.readFileSync(inputPath, "utf-8");
if (maxChars > 0 && text.length > maxChars) {
  text = text.slice(0, maxChars);
  console.log(`Truncated:  ${(maxChars / 1e6).toFixed(0)}MB of ${(text.length / 1e6).toFixed(0)}MB`);
}
console.log(`Corpus:     ${(text.length / 1024).toFixed(0)}KB (${text.length} bytes)\n`);

console.log("Training BPE merges...");
const startTime = Date.now();

const bpe = cuda.bpeCreate();
cuda.bpeTrain(bpe, text, vocabSize);

const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
const finalSize = cuda.bpeVocabSize(bpe);
console.log(`Done in ${elapsed}s — vocab size: ${finalSize}\n`);

// Show a few example encodings
const examples = ["the ", "Once upon a time", "Hello, world!", "She was very happy"];
console.log("Example encodings:");
for (const ex of examples) {
  const tokens = Array.from(cuda.bpeEncode(bpe, ex));
  const decoded = cuda.bpeDecode(bpe, new Int32Array(tokens));
  console.log(`  "${ex}" -> [${tokens.join(", ")}] (${tokens.length} tokens) -> "${decoded}"`);
}

cuda.bpeSave(bpe, outputPath);
console.log(`\nVocab saved to ${outputPath}`);
cuda.bpeFree(bpe);
