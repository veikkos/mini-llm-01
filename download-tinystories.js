// Download TinyStories dataset from HuggingFace
// Creates data/tinystories.txt
//
// Run: node download-tinystories.js
// Run: node download-tinystories.js --size 1000

const fs = require("fs");
const path = require("path");
const https = require("https");

const args = process.argv.slice(2);
const getArg = (name, def) => {
  const i = args.indexOf("--" + name);
  return i >= 0 ? args[i + 1] : def;
};

const OUT_DIR = path.join(__dirname, "data");
const OUT_FILE = path.join(OUT_DIR, "tinystories.txt");
const sizeMB = parseInt(getArg("size", "250"));
const MAX_BYTES = sizeMB * 1_000_000;

// HuggingFace direct download URL for the training text file
const URL =
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt";

function downloadToFile(url, outPath, maxBytes) {
  return new Promise((resolve, reject) => {
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    const tmpPath = outPath + ".tmp";
    const fileStream = fs.createWriteStream(tmpPath);
    let totalBytes = 0;

    const request = (u) => {
      https.get(u, { headers: { "User-Agent": "node" } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          request(res.headers.location);
          return;
        }
        if (res.statusCode !== 200) {
          fileStream.close();
          fs.unlinkSync(tmpPath);
          reject(new Error(`HTTP ${res.statusCode} from ${u}`));
          return;
        }

        res.on("data", (chunk) => {
          totalBytes += chunk.length;
          fileStream.write(chunk);
          process.stdout.write(`\r  Downloading: ${(totalBytes / 1e6).toFixed(1)}MB / ${(maxBytes / 1e6).toFixed(0)}MB`);

          if (totalBytes >= maxBytes) {
            res.destroy();
            console.log("  Done.");
            fileStream.end(() => resolve({ tmpPath, totalBytes }));
          }
        });

        res.on("end", () => {
          console.log("  Done.");
          fileStream.end(() => resolve({ tmpPath, totalBytes }));
        });

        res.on("error", (err) => {
          if (totalBytes >= maxBytes) return;
          fileStream.close();
          fs.unlinkSync(tmpPath);
          reject(err);
        });
      });
    };
    request(url);
  });
}

async function main() {
  if (fs.existsSync(OUT_FILE)) {
    const fileSizeMB = (fs.statSync(OUT_FILE).size / 1e6).toFixed(1);
    console.log(`Already exists: ${OUT_FILE} (${fileSizeMB}MB)`);
    console.log("Delete it to re-download.");
    return;
  }

  console.log("=== Downloading TinyStories ===\n");
  console.log("Source: HuggingFace roneneldan/TinyStories");
  console.log(`Target: ${OUT_FILE}`);
  console.log(`Downloading first ~${(MAX_BYTES / 1e6).toFixed(0)}MB (of ~1.9GB total)\n`);

  const { tmpPath, totalBytes } = await downloadToFile(URL, OUT_FILE, MAX_BYTES);
  console.log(`  Downloaded: ${(totalBytes / 1e6).toFixed(1)}MB`);

  // Stream-process: clean up markers, trim at story boundary
  process.stdout.write("  Processing...");
  const CHUNK = 16 * 1024 * 1024; // 16MB read chunks
  const readStream = fs.createReadStream(tmpPath, { encoding: "utf-8", highWaterMark: CHUNK });
  const writeStream = fs.createWriteStream(OUT_FILE);
  let leftover = "";
  let storyCount = 0;

  await new Promise((resolve, reject) => {
    readStream.on("data", (chunk) => {
      let data = leftover + chunk;
      data = data.replace(/<\|endoftext\|>/g, "\n\n");
      data = data.replace(/\n{3,}/g, "\n\n");

      // Keep last few bytes as leftover in case a marker spans chunks
      const keep = 20;
      leftover = data.slice(-keep);
      const toWrite = data.slice(0, -keep);
      storyCount += (toWrite.match(/\n\n/g) || []).length;
      writeStream.write(toWrite);
    });
    readStream.on("end", () => {
      leftover = leftover.replace(/<\|endoftext\|>/g, "\n\n");
      leftover = leftover.replace(/\n{3,}/g, "\n\n");
      leftover = leftover.trimEnd() + "\n";
      writeStream.write(leftover);
      writeStream.end(resolve);
    });
    readStream.on("error", reject);
  });

  const finalSize = fs.statSync(OUT_FILE).size;
  console.log(` done.`);
  console.log(`\n  Saved: ${(finalSize / 1e6).toFixed(1)}MB, ~${storyCount.toLocaleString()} stories`);

  // Clean up temp file
  fs.unlinkSync(tmpPath);
  process.exit(0);
}

main().catch((e) => {
  console.error("Error:", e.message);
  process.exit(1);
});
