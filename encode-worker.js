// Worker thread for parallel BPE encoding
const { parentPort, workerData } = require("worker_threads");
const { cuda } = require("./gpu-tensor");

const bpe = cuda.bpeCreate();
cuda.bpeLoad(bpe, workerData.vocabPath);

parentPort.on("message", (msg) => {
  if (msg.type === "encode") {
    const tokens = cuda.bpeEncode(bpe, msg.text);
    parentPort.postMessage({ type: "result", index: msg.index, tokens });
  } else if (msg.type === "exit") {
    cuda.bpeFree(bpe);
    process.exit(0);
  }
});

parentPort.postMessage({ type: "ready" });
