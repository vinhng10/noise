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

    for (let i = 0; i < input[0].length; i++) {
      this.mono[i] = (input[0][i] + input[1][i]) / 2;
      output[0][i] = output[1][i] = this.output[i];
    }

    this.workerPort.postMessage(this.mono);

    return true;
  }
}

registerProcessor("processor", Processor);
