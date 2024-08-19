import * as ort from "onnxruntime-web/webgpu";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// Use the Singleton pattern to enable lazy construction of the pipeline.
class PipelineSingleton {
  static session = null;
  static loading = false;
  static input = new Float32Array(48000);

  static async getSession() {
    if (this.session === null && !this.loading) {
      this.loading = true;
      const opt = {
        executionProviders: ["webgpu"],
      };
      const url = new URL("./model.onnx", import.meta.url);
      this.session = await ort.InferenceSession.create(url.toString(), opt);
      this.loading = false;
    }
    return this.session;
  }

  static getInput() {
    return this.input;
  }

  static updateInput(newInput) {
    const input = new Float32Array(48000);
    const oldInput = this.input.slice(128);
    input.set(oldInput, 0);
    input.set(newInput, oldInput.length);
    this.input = input;
    return this.input;
  }
}

self.onmessage = async (event) => {
  const { port } = event.data;
  let session = await PipelineSingleton.getSession();

  // Listen for messages on the shared port
  port.onmessage = async (event) => {
    PipelineSingleton.updateInput(event.data);
  };

  while (true) {
    const tensor = new ort.Tensor(
      "float32",
      PipelineSingleton.getInput(),
      [1, 1, 1, 48000]
    );
    const result = await session.run({ input: tensor });
    port.postMessage(result.output.data.slice(48000 - 128));
  }
};
