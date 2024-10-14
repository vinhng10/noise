import * as ort from "onnxruntime-web/webgpu";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let cutoff = 100;
let maxSize = 300;
let size = 50;

// Returns a low-pass transform function for use with TransformStream.
function noiseFilter(session) {
  const format = "f32-planar";
  let lastValuePerChannel = undefined;
  let buffer = undefined;
  let rc = 1.0 / (cutoff * 2 * Math.PI);
  let dt = undefined;
  let alpha = undefined;
  let frames = undefined;
  let offset = 0;
  let outputs = [];

  return async (data, controller) => {
    if (!lastValuePerChannel) {
      console.log(`Audio stream has ${data.numberOfChannels} channels.`);
      lastValuePerChannel = Array(1).fill(0);
    }
    if (!frames) frames = data.numberOfFrames;
    if (!buffer) buffer = new Float32Array(frames * maxSize);
    if (!dt) dt = 1.0 / data.sampleRate;
    if (!alpha) alpha = dt / (rc + dt);

    // Extract audio data from input:
    const samples = buffer.subarray(
      (maxSize - size + offset) * frames,
      (maxSize - size + offset + 1) * frames
    );
    data.copyTo(samples, { planeIndex: 0, format });
    offset += 1;

    // Apply low-pass filter to samples:
    let lastValue = lastValuePerChannel[0];
    for (let i = 0; i < samples.length; ++i) {
      lastValue = lastValue + alpha * (samples[i] - lastValue);
      samples[i] = lastValue;
    }
    lastValuePerChannel[0] = lastValue;

    // Run audio processing:
    if (offset === size) {
      const tensor = new ort.Tensor("float32", buffer, [
        1,
        1,
        1,
        buffer.length,
      ]);
      const result = await session.run({ input: tensor });
      console.log(result.output.data.length);
      const output = result.output.data.subarray((maxSize - size) * frames);
      // const output = buffer.subarray((maxSize - size) * frames);
      for (let i = 0; i < output.length; i += frames) {
        outputs.push(output.subarray(i, i + frames));
      }
      buffer
        .subarray(0, (maxSize - size) * frames)
        .set(buffer.subarray(size * frames));
      offset = 0;
    }

    // Return the processed audio:
    if (outputs.length <= 0) return;

    controller.enqueue(
      new AudioData({
        format,
        sampleRate: data.sampleRate,
        numberOfFrames: frames,
        numberOfChannels: data.numberOfChannels,
        timestamp: data.timestamp,
        data: outputs.shift(),
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
      transform: noiseFilter(session),
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
