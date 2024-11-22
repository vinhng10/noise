import "./App.css";
import { useState, useRef, useEffect } from "react";

function App() {
  const [isAudioOn, setIsAudioOn] = useState(false);
  const [isNoiseReductionEnabled, setIsNoiseReductionEnabled] = useState(false);
  const stream = useRef(null);
  const audioContext = useRef(null);
  const analyser = useRef(null);
  const canvasRef = useRef(null);
  const waveformCanvasRef = useRef(null);
  const animationId = useRef(null);
  const mediaRecorder = useRef(null);
  const recordedChunks = useRef([]);

  const drawSpectrogram = () => {
    if (!analyser.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext("2d");
    const bufferLength = analyser.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      analyser.current.getByteFrequencyData(dataArray);
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;

      dataArray.forEach((barHeight, i) => {
        const r = barHeight + 25 * (i / bufferLength);
        const g = 250 * (i / bufferLength);
        const b = 50;
        canvasCtx.fillStyle = `rgb(${r},${g},${b})`;
        canvasCtx.fillRect(
          x,
          canvas.height - barHeight / 2,
          barWidth,
          barHeight / 2
        );
        x += barWidth + 1;
      });

      animationId.current = requestAnimationFrame(draw);
    };

    draw();
  };

  const drawWaveform = () => {
    if (!analyser.current || !waveformCanvasRef.current) return;
    const canvas = waveformCanvasRef.current;
    const canvasCtx = canvas.getContext("2d");
    const bufferLength = analyser.current.fftSize;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      analyser.current.getByteTimeDomainData(dataArray);
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = "rgb(0, 0, 0)";
      canvasCtx.beginPath();

      const sliceWidth = canvas.width / bufferLength;
      let x = 0;

      dataArray.forEach((v, i) => {
        const y = (v / 128.0) * (canvas.height / 2);
        i === 0 ? canvasCtx.moveTo(x, y) : canvasCtx.lineTo(x, y);
        x += sliceWidth;
      });

      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
      animationId.current = requestAnimationFrame(draw);
    };

    draw();
  };

  const startAudio = async () => {
    try {
      setIsAudioOn(true);
      stream.current = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      audioContext.current = new AudioContext({ sampleRate: 16000 });

      const resampledSource = audioContext.current.createMediaStreamSource(
        stream.current
      );
      const resampledSink = audioContext.current.createMediaStreamDestination();
      resampledSource.connect(resampledSink);

      const track = resampledSink.stream.getAudioTracks()[0];
      const processor = new MediaStreamTrackProcessor(track);
      const generator = new MediaStreamTrackGenerator("audio");

      const source = processor.readable;
      const sink = generator.writable;

      const worker = new Worker(new URL("./worker.js", import.meta.url), {
        type: "module",
      });
      worker.postMessage({ source, sink }, [source, sink]);

      const processedStream = new MediaStream([generator]);
      const processedSource =
        audioContext.current.createMediaStreamSource(processedStream);
      processedSource.connect(audioContext.current.destination);

      analyser.current = audioContext.current.createAnalyser();
      processedSource.connect(analyser.current);

      drawSpectrogram();
      drawWaveform();
    } catch (error) {
      setIsAudioOn(false);
      console.error(error);
    }
  };

  const stopAudio = async () => {
    setIsAudioOn(false);
    stream.current.getTracks().forEach((track) => track.stop());
    audioContext.current.close();

    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
      mediaRecorder.current.onstop = async () => {
        const blob = new Blob(recordedChunks.current, { type: "audio/webm" });
        recordedChunks.current = [];
        const audioURL = URL.createObjectURL(blob);
        const audio = new Audio(audioURL);
        audio.controls = true;
        document.body.appendChild(audio);
      };
    }

    if (animationId.current) {
      cancelAnimationFrame(animationId.current);
    }
  };

  useEffect(() => {
    return () => {
      if (animationId.current) {
        cancelAnimationFrame(animationId.current);
      }
    };
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <div className="mt-8">
          <button
            className="btn start-btn"
            onClick={startAudio}
            disabled={isAudioOn}
          >
            Start
          </button>
          <button
            className="btn stop-btn"
            onClick={stopAudio}
            disabled={!isAudioOn}
          >
            Stop
          </button>
          <button
            className="btn"
            onClick={() => setIsNoiseReductionEnabled(!isNoiseReductionEnabled)}
            disabled={isAudioOn}
          >
            Noise Reduction: {isNoiseReductionEnabled ? "ON" : "OFF"}
          </button>
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "20px",
            marginTop: "20px",
          }}
        >
          <canvas
            ref={canvasRef}
            width="600"
            height="200"
            style={{ border: "1px solid black" }}
          />
          <canvas
            ref={waveformCanvasRef}
            width="600"
            height="200"
            style={{ border: "1px solid black" }}
          />
        </div>
      </header>
    </div>
  );
}

export default App;
