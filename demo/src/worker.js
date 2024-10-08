import * as ort from "onnxruntime-web/webgpu";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let cutoff = 100;
let size = 10;

// Returns a low-pass transform function for use with TransformStream.
function lowPassFilter(session) {
  const format = "f32-planar";
  let lastValuePerChannel = undefined;
  let buffer = undefined;
  let rc = 1.0 / (cutoff * 2 * Math.PI);
  let dt = undefined;
  let alpha = undefined;
  let offset = 0;
  let outputs = [];
  let lastOutput = undefined;

  return async (data, controller) => {
    if (!lastValuePerChannel) {
      console.log(`Audio stream has ${data.numberOfChannels} channels.`);
      lastValuePerChannel = Array(1).fill(0);
    }
    if (!buffer) buffer = new Float32Array(data.numberOfFrames * size);
    if (!lastOutput) lastOutput = new Float32Array(data.numberOfFrames);
    if (!dt) dt = 1.0 / data.sampleRate;
    if (!alpha) alpha = dt / (rc + dt);

    const samples = buffer.subarray(
      offset * data.numberOfFrames,
      (offset + 1) * data.numberOfFrames
    );
    data.copyTo(samples, { planeIndex: 0, format });
    let lastValue = lastValuePerChannel[0];

    // Apply low-pass filter to samples.
    for (let i = 0; i < samples.length; ++i) {
      lastValue = lastValue + alpha * (samples[i] - lastValue);
      samples[i] = lastValue;
    }

    lastValuePerChannel[0] = lastValue;
    offset += 1;

    if (offset === size) {
      console.log(offset);
      const tensor = new ort.Tensor("float32", buffer, [
        1,
        1,
        1,
        buffer.length,
      ]);
      const result = await session.run({ input: tensor });
      const output = result.output.data;
      for (let i = 0; i < output.length; i += data.numberOfFrames) {
        outputs.push(output.subarray(i, i + data.numberOfFrames));
      }
      offset = 0;
    }

    if (outputs.length > 0) lastOutput = outputs.shift();

    controller.enqueue(
      new AudioData({
        format,
        sampleRate: data.sampleRate,
        numberOfFrames: data.numberOfFrames,
        numberOfChannels: data.numberOfChannels,
        timestamp: data.timestamp,
        data: lastOutput,
      })
    );
  };
}

let abortController;

onmessage = async (event) => {
  if (event.data.command === "abort") {
    abortController.abort();
    abortController = null;
  } else {
    const url = new URL("./model.onnx", import.meta.url);
    const session = await ort.InferenceSession.create(url.toString(), {
      executionProviders: ["webgpu"],
    });
    console.log(session);
    const source = event.data.source;
    const sink = event.data.sink;
    const transformer = new TransformStream({
      transform: lowPassFilter(session),
    });
    abortController = new AbortController();
    const signal = abortController.signal;
    const promise = source.pipeThrough(transformer, { signal }).pipeTo(sink);
    promise.catch((e) => {
      if (signal.aborted) {
        console.log("Shutting down streams after abort.");
      } else {
        console.error("Error from stream transform:", e);
      }
      source.cancel(e);
      sink.abort(e);
    });
  }
};
