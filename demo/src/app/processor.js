class Processor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.output = new Float32Array(128);
    this.mono = new Float32Array(128);

    this.port.onmessage = (event) => {
      const { port } = event.data;
      this.workerPort = port;
      this.workerPort.onmessage = (event) => {
        this.output = event.data;
      };
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    output[0].set(this.output);
    output[1].set(this.output);

    this.workerPort.postMessage(input[0]);

    return true;
  }
}

registerProcessor("processor", Processor);
