import * as ort from "onnxruntime-web/webgpu";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let session = null;
let loading = false;
let input = new Float32Array(4096);

let getSession = async () => {
  if (session === null && !loading) {
    loading = true;
    const url = new URL("./model.onnx", import.meta.url);
    session = await ort.InferenceSession.create(url.toString(), {
      executionProviders: ["webgpu"],
    });
    loading = false;
  }
};

let updateInput = (newInput) => {
  const tmp = new Float32Array(4096);
  const oldInput = input.slice(128);
  tmp.set(oldInput, 0);
  tmp.set(newInput, oldInput.length);
  input = tmp;
  return input;
};

self.onmessage = async (event) => {
  const { port } = event.data;
  await getSession();

  // Listen for messages on the shared port
  port.onmessage = async (event) => {
    updateInput(event.data);
  };

  while (true) {
    const tensor = new ort.Tensor("float32", input, [1, 1, 1, 4096]);
    const result = await session.run({ input: tensor });
    port.postMessage(result.output.data.slice(4096 - 128));
  }
};
