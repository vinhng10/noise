import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const maxSize = 50;
const size = 20;

// Returns a low-pass transform function for use with TransformStream.
const noiseFilter = (session) => {
  const format = "f32-planar";
  let frames;
  let offset = 0;
  let buffer;
  let outputs = [];

  return async (data, controller) => {
    if (!frames) frames = data.numberOfFrames;
    if (!buffer) buffer = new Float32Array(frames * maxSize);

    // Extract audio data from input:
    const samples = buffer.subarray(
      (maxSize - size + offset) * frames,
      (maxSize - size + offset + 1) * frames
    );
    data.copyTo(samples, { planeIndex: 0, format });
    offset += 1;

    // Run audio processing:
    if (offset === size) {
      const tensor = new ort.Tensor("float32", buffer, [
        1,
        1,
        1,
        buffer.length,
      ]);

      const startTime = performance.now(); // Start time
      const result = await session.run({ input: tensor });
      const endTime = performance.now(); // End time
      console.log(`session.run() took ${endTime - startTime} ms`);

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
        numberOfChannels: 1,
        timestamp: data.timestamp,
        data: audioData,
      })
    );
  };
};

onmessage = async (event) => {
  const url = new URL("./model.onnx", import.meta.url);
  const session = await ort.InferenceSession.create(url.toString(), {
    executionProviders: ["wasm"],
  });
  const source = event.data.source;
  const sink = event.data.sink;
  const transformer = new TransformStream({
    transform: noiseFilter(session),
  });
  source.pipeThrough(transformer).pipeTo(sink);
};
