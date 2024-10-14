import "./App.css";
import { useState, useRef } from "react";

function App() {
  const [isAudioOn, setIsAudioOn] = useState(false);
  const stream = useRef(null);
  const processor = useRef(null);
  const generator = useRef(null);
  const worker = useRef(null);
  const processedStream = useRef(null);
  const audioContext = useRef(null);

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

      audioContext.current = new AudioContext();
      const sourceNode = audioContext.current.createMediaStreamSource(
        processedStream.current
      );
      sourceNode.connect(audioContext.current.destination);
    } catch (error) {
      setIsAudioOn(false);
      console.error(error);
    }
  };

  const stopAudio = async () => {
    setIsAudioOn(false);
    stream.current.getTracks().forEach((track) => track.stop());
    worker.current.postMessage({ command: "abort" });
    worker.current.terminate();
    audioContext.current.close();
  };

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
        </div>
      </header>
    </div>
  );
}

export default App;
