import * as ort from "onnxruntime-web/webgpu";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let cutoff = 100;

// Returns a low-pass transform function for use with TransformStream.
function lowPassFilter(session) {
  const format = "f32-planar";
  let lastValuePerChannel = undefined;

  return async (data, controller) => {
    const rc = 1.0 / (cutoff * 2 * Math.PI);
    const dt = 1.0 / data.sampleRate;
    const alpha = dt / (rc + dt);
    const nChannels = data.numberOfChannels;
    if (!lastValuePerChannel) {
      console.log(`Audio stream has ${nChannels} channels.`);
      lastValuePerChannel = Array(nChannels).fill(0);
    }
    const buffer = new Float32Array(data.numberOfFrames * nChannels * 10);
    for (let c = 0; c < nChannels; c++) {
      const offset = data.numberOfFrames * c;
      const samples = buffer.subarray(offset, offset + data.numberOfFrames);
      data.copyTo(samples, { planeIndex: c, format });
      let lastValue = lastValuePerChannel[c];

      // Apply low-pass filter to samples.
      for (let i = 0; i < samples.length; ++i) {
        lastValue = lastValue + alpha * (samples[i] - lastValue);
        samples[i] = lastValue;
      }

      lastValuePerChannel[c] = lastValue;

      const tensor = new ort.Tensor("float32", buffer, [1, 1, 1, 4800]);
      const result = await session.run({ input: tensor });
      samples.set(result.output.data.slice(0, 480));
    }
    controller.enqueue(
      new AudioData({
        format,
        sampleRate: data.sampleRate,
        numberOfFrames: data.numberOfFrames,
        numberOfChannels: nChannels,
        timestamp: data.timestamp,
        data: buffer.slice(0, 480),
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
