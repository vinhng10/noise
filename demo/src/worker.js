import * as ort from "onnxruntime-web/webgpu";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

const cutoff = 500;
const maxSize = 300;
const size = 30;

// Returns a low-pass transform function for use with TransformStream.
const noiseFilter = (session) => {
  const format = "f32-planar";
  let lastValuePerChannel = 0;
  const rc = 1.0 / (cutoff * 2 * Math.PI);
  let dt;
  let alpha;
  let frames;
  let offset = 0;
  let buffer;
  let outputs = [];

  return async (data, controller) => {
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
    let lastValue = lastValuePerChannel;
    for (let i = 0; i < samples.length; ++i) {
      lastValue = lastValue + alpha * (samples[i] - lastValue);
      samples[i] = lastValue;
    }
    lastValuePerChannel = lastValue;

    // Run audio processing:
    if (offset === size) {
      const tensor = new ort.Tensor("float32", buffer, [
        1,
        1,
        1,
        buffer.length,
      ]);
      const result = await session.run({ input: tensor });
      const output = result.output.data.subarray((maxSize - size) * frames);
      for (let i = 0; i < output.length; i += frames) {
        outputs.push(output.subarray(i, i + frames));
      }
      buffer
        .subarray(0, (maxSize - size) * frames)
        .set(buffer.subarray(size * frames));
      offset = 0;
    }

    // Return the processed audio:
    const audioData = outputs.shift();
    if (audioData == undefined) return;

    controller.enqueue(
      new AudioData({
        format,
        sampleRate: data.sampleRate,
        numberOfFrames: frames,
        numberOfChannels: data.numberOfChannels,
        timestamp: data.timestamp,
        data: audioData,
      })
    );
  };
};

onmessage = async (event) => {
  const url = new URL("./model.onnx", import.meta.url);
  const session = await ort.InferenceSession.create(url.toString(), {
    executionProviders: ["webgpu"],
  });
  const source = event.data.source;
  const sink = event.data.sink;
  const transformer = new TransformStream({
    transform: noiseFilter(session),
  });
  source.pipeThrough(transformer).pipeTo(sink);
};
