import "./App.css";
import { useState, useRef, useEffect } from "react";

function App() {
  const [isAudioOn, setIsAudioOn] = useState(false);
  const [isNoiseReductionEnabled, setIsNoiseReductionEnabled] = useState(false);
  const stream = useRef(null);
  const processor = useRef(null);
  const generator = useRef(null);
  const worker = useRef(null);
  const processedStream = useRef(null);
  const audioContext = useRef(null);
  const analyser = useRef(null);
  const canvasRef = useRef(null); // Spectrogram canvas
  const waveformCanvasRef = useRef(null); // Waveform canvas
  const animationId = useRef(null);

  // Function to draw the spectrogram (frequency domain)
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
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i];

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
      }

      animationId.current = requestAnimationFrame(draw);
    };

    draw();
  };

  // Function to draw the waveform (time domain)
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

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0; // Normalize the value
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          canvasCtx.moveTo(x, y);
        } else {
          canvasCtx.lineTo(x, y);
        }

        x += sliceWidth;
      }

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
        video: false,
      });
      const track = stream.current.getAudioTracks()[0];
      console.log("Using audio device: " + track.label);
      stream.current.oninactive = () => {
        console.log("Stream ended");
      };

      audioContext.current = new AudioContext();
      const sourceNode = audioContext.current.createMediaStreamSource(
        stream.current
      );

      // Create Analyser Node
      analyser.current = audioContext.current.createAnalyser();
      analyser.current.fftSize = 2048;

      const filterNode = audioContext.current.createBiquadFilter();
      filterNode.type = "bandpass";
      filterNode.frequency.value = 2500; // Center frequency
      filterNode.Q.value = 5; // Quality factor to narrow the range

      const gainNode = audioContext.current.createGain();
      gainNode.gain.value = 1;

      if (isNoiseReductionEnabled) {
        processor.current = new MediaStreamTrackProcessor(track);
        generator.current = new MediaStreamTrackGenerator("audio");
        const source = processor.current.readable;
        const sink = generator.current.writable;
        worker.current = new Worker(new URL("./worker.js", import.meta.url), {
          type: "module",
        });
        worker.current.postMessage({ source: source, sink: sink }, [
          source,
          sink,
        ]);

        processedStream.current = new MediaStream();
        processedStream.current.addTrack(generator.current);

        const processedSourceNode =
          audioContext.current.createMediaStreamSource(processedStream.current);
        processedSourceNode.connect(analyser.current);
      } else {
        sourceNode.connect(analyser.current);
      }

      analyser.current.connect(audioContext.current.destination);

      // Start drawing both spectrogram and waveform
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
    if (isNoiseReductionEnabled) {
      worker.current.postMessage({ command: "abort" });
      worker.current.terminate();
    }
    audioContext.current.close();

    // Cancel the animation
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
          {/* Canvas for Spectrogram */}
          <canvas
            ref={canvasRef}
            width="600"
            height="200"
            style={{ border: "1px solid black" }}
          />

          {/* Canvas for Waveform */}
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
