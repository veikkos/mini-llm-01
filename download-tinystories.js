// Download TinyStories dataset from HuggingFace
// Creates data/tinystories.txt with ~10M tokens worth of stories
//
// Run: node download-tinystories.js

const fs = require("fs");
const path = require("path");
const https = require("https");

const OUT_DIR = path.join(__dirname, "data");
const OUT_FILE = path.join(OUT_DIR, "tinystories.txt");
const MAX_BYTES = 250_000_000; // download ~250MB then trim at story boundary

// HuggingFace direct download URL for the training text file
const URL =
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt";

function downloadPartial(url, maxBytes) {
  return new Promise((resolve, reject) => {
    const request = (u) => {
      https.get(u, { headers: { "User-Agent": "node" } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          request(res.headers.location);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} from ${u}`));
          return;
        }

        const chunks = [];
        let totalBytes = 0;

        res.on("data", (chunk) => {
          chunks.push(chunk);
          totalBytes += chunk.length;
          process.stdout.write(`\r  Downloading: ${(totalBytes / 1e6).toFixed(1)}MB / ${(maxBytes / 1e6).toFixed(0)}MB`);

          if (totalBytes >= maxBytes) {
            res.destroy(); // stop downloading
            console.log("  Done.");
            resolve(Buffer.concat(chunks).toString("utf-8"));
          }
        });

        res.on("end", () => {
          console.log("  Done.");
          resolve(Buffer.concat(chunks).toString("utf-8"));
        });

        res.on("error", (err) => {
          // Ignore error from our res.destroy()
          if (totalBytes >= maxBytes) return;
          reject(err);
        });
      });
    };
    request(url);
  });
}

async function main() {
  if (fs.existsSync(OUT_FILE)) {
    const sizeMB = (fs.statSync(OUT_FILE).size / 1e6).toFixed(1);
    console.log(`Already exists: ${OUT_FILE} (${sizeMB}MB)`);
    console.log("Delete it to re-download.");
    return;
  }

  console.log("=== Downloading TinyStories ===\n");
  console.log("Source: HuggingFace roneneldan/TinyStories");
  console.log(`Target: ${OUT_FILE}`);
  console.log(`Downloading first ~${(MAX_BYTES / 1e6).toFixed(0)}MB (of ~1.9GB total)\n`);

  const raw = await downloadPartial(URL, MAX_BYTES);
  console.log(`  Downloaded: ${(raw.length / 1e6).toFixed(1)}MB`);

  // Cut at a story boundary (stories separated by <|endoftext|>)
  let text = raw;
  const cutPoint = text.lastIndexOf("<|endoftext|>");
  if (cutPoint > 0) {
    text = text.slice(0, cutPoint + "<|endoftext|>".length);
  }

  // Clean up: remove <|endoftext|> markers, collapse whitespace between stories
  text = text.replace(/<\|endoftext\|>/g, "\n\n");
  text = text.replace(/\n{3,}/g, "\n\n");
  text = text.trim() + "\n";

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_FILE, text);
  const stories = text.split("\n\n").length;
  console.log(`\n  Saved: ${(text.length / 1e6).toFixed(1)}MB, ~${stories.toLocaleString()} stories`);
  process.exit(0);
}

main().catch((e) => {
  console.error("Error:", e.message);
  process.exit(1);
});
