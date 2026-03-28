// Pre-encode text into a binary token file for fast training
// Run: node encode.js --input data/tinystories.txt
// Run: node encode.js --input data/tinystories.txt --output tokens.bin

const fs = require("fs");
const path = require("path");
const os = require("os");
const { Worker } = require("worker_threads");
const { cuda } = require("./gpu-tensor");

const args = process.argv.slice(2);
const getArg = (name, def) => {
  const i = args.indexOf("--" + name);
  return i >= 0 ? args[i + 1] : def;
};

const inputPath = path.resolve(getArg("input", path.join(__dirname, "data/tinystories.txt")));
const vocabPath = path.resolve(getArg("vocab", path.join(__dirname, "vocab.json")));
const outputPath = path.resolve(getArg("output", path.join(__dirname, "tokens.bin")));
const numWorkers = parseInt(getArg("workers", String(Math.max(1, os.cpus().length - 1))));

console.log("=== BPE Encode ===\n");

if (!fs.existsSync(vocabPath)) {
  console.error("No vocab.json found. Train vocabulary first:\n  node train-vocab.js\n");
  process.exit(1);
}

const bpe = cuda.bpeCreate();
cuda.bpeLoad(bpe, vocabPath);
const vocabSize = cuda.bpeVocabSize(bpe);
cuda.bpeFree(bpe);
console.log(`Vocab: ${vocabSize} tokens`);

const maxChars = parseInt(getArg("maxchars", "0"));
const CHUNK_SIZE = 1_000_000; // 1MB per chunk

// Read file in chunks to avoid Node.js string length limit
process.stdout.write(`Loading ${path.basename(inputPath)}...`);
const fileSize = fs.statSync(inputPath).size;
const readSize = maxChars > 0 ? Math.min(maxChars, fileSize) : fileSize;
const chunks = [];
const fd = fs.openSync(inputPath, "r");
let bytesRead = 0;
while (bytesRead < readSize) {
  const toRead = Math.min(CHUNK_SIZE, readSize - bytesRead);
  const buf = Buffer.alloc(toRead);
  fs.readSync(fd, buf, 0, toRead, bytesRead);
  chunks.push(buf.toString("utf-8"));
  bytesRead += toRead;
}
fs.closeSync(fd);
console.log(` ${(bytesRead / 1e6).toFixed(1)}MB`);
if (maxChars > 0 && maxChars < fileSize) {
  console.log(`Truncated to ${(maxChars / 1e6).toFixed(0)}MB`);
}

const actualWorkers = Math.min(numWorkers, chunks.length);
console.log(`Encoding ${chunks.length} chunks with ${actualWorkers} workers...`);
const start = Date.now();

// Results array indexed by chunk order
const results = new Array(chunks.length);
let chunksCompleted = 0;
let nextChunk = 0;

function spawnWorker() {
  return new Promise((resolve) => {
    const worker = new Worker(path.join(__dirname, "encode-worker.js"), {
      workerData: { vocabPath },
    });

    worker.on("message", (msg) => {
      if (msg.type === "ready") {
        sendNext(worker);
      } else if (msg.type === "result") {
        results[msg.index] = msg.tokens;
        chunksCompleted++;

        const elapsedS = (Date.now() - start) / 1000;
        const etaS = (elapsedS / chunksCompleted) * (chunks.length - chunksCompleted);
        process.stdout.write(
          `\r  Chunk ${chunksCompleted}/${chunks.length} | ${elapsedS.toFixed(0)}s elapsed | ~${etaS.toFixed(0)}s remaining`
        );

        sendNext(worker);
      }
    });

    function sendNext(w) {
      if (nextChunk < chunks.length) {
        const idx = nextChunk++;
        w.postMessage({ type: "encode", index: idx, text: chunks[idx] });
      } else {
        w.postMessage({ type: "exit" });
        resolve();
      }
    }
  });
}

Promise.all(Array.from({ length: actualWorkers }, () => spawnWorker())).then(() => {
  console.log();

  // Merge chunks into single Int32Array
  const totalLen = results.reduce((s, t) => s + t.length, 0);
  const tokens = new Int32Array(totalLen);
  let offset = 0;
  for (const t of results) {
    tokens.set(t, offset);
    offset += t.length;
  }

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log(`Encoded ${tokens.length.toLocaleString()} tokens in ${elapsed}s`);
  console.log(`Compression: ${(tokens.length > 0 ? chunks.reduce((s, c) => s + c.length, 0) / tokens.length : 0).toFixed(1)} chars/token`);

  // Save as raw Int32Array binary
  const buf = Buffer.from(tokens.buffer, tokens.byteOffset, tokens.byteLength);
  fs.writeFileSync(outputPath, buf);
  const sizeMB = (buf.length / 1e6).toFixed(1);
  console.log(`\nSaved: ${outputPath} (${sizeMB}MB)`);
  console.log(`\nTrain with:\n  node train.js --tokens ${path.basename(outputPath)}`);
});
