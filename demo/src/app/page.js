"use client";

import { useState, useEffect, useRef, useCallback } from "react";

export default function Home() {
  // Keep track of the classification result and the model loading status.
  const [isAudioOn, setIsAudioOn] = useState(false);
  const [result, setResult] = useState(null);
  const [ready, setReady] = useState(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const worker = useRef(null);

  const startAudio = async () => {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter.features.has("shader-f16")) {
        console.log("====> shader-f16");
      }

      // Create a new MessageChannel
      const messageChannel = new MessageChannel();

      // Create the web worker and pass one end of the port to it
      worker.current = new Worker(new URL("./worker.js", import.meta.url), {
        type: "module",
      });
      worker.current.postMessage({ port: messageChannel.port1 }, [
        messageChannel.port1,
      ]);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new AudioContext({
        sampleRate: 16000,
        echoCancellation: true,
        autoGainControl: true,
        noiseSuppression: true,
      });
      audioContextRef.current = audioContext;

      // Load the AudioWorkletProcessor script
      await audioContext.audioWorklet.addModule(
        new URL("./processor.js", import.meta.url)
      );
      // Create the AudioWorkletNode
      const audioWorkletNode = new AudioWorkletNode(audioContext, "processor");

      // Pass the other end of the port to the audio worklet
      audioWorkletNode.port.postMessage({ port: messageChannel.port2 }, [
        messageChannel.port2,
      ]);

      const source = audioContext.createMediaStreamSource(stream);
      const destination = audioContext.destination;
      source.connect(audioWorkletNode);
      audioWorkletNode.connect(destination);

      setIsAudioOn(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const stopAudio = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    worker.current.terminate();
    setIsAudioOn(false);
  };

  // We use the `useEffect` hook to set up the worker as soon as the `App` component is mounted.
  useEffect(() => {
    if (!worker.current) {
      // Create the worker if it does not yet exist.
      worker.current = new Worker(new URL("./worker.js", import.meta.url), {
        type: "module",
      });
    }
    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "initiate":
          setReady(false);
          break;
        case "ready":
          setReady(true);
          break;
        case "complete":
          setResult(e.data.output[0]);
          break;
      }
    };
    // Attach the callback function as an event listener.
    worker.current.addEventListener("message", onMessageReceived);
    // Define a cleanup function for when the component is unmounted.
    return () =>
      worker.current.removeEventListener("message", onMessageReceived);
  }, []);

  const classify = useCallback((text) => {
    if (worker.current) {
      worker.current.postMessage({ text });
    }
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-12">
      <h1 className="text-5xl font-bold mb-2 text-center">
        Noise Suppression ???
      </h1>
      <div className="mb-4">
        <button
          className={`px-8 py-4 text-lg rounded mr-4 ${
            isAudioOn ? "bg-gray-300" : "bg-green-500 text-white"
          }`}
          onClick={startAudio}
          disabled={isAudioOn}
        >
          On
        </button>
        <button
          className={`px-8 py-4 text-lg rounded ${
            !isAudioOn ? "bg-gray-300" : "bg-red-500 text-white"
          }`}
          onClick={stopAudio}
          disabled={!isAudioOn}
        >
          Off
        </button>
      </div>

      {ready !== null && (
        <pre className="bg-gray-100 p-2 rounded">
          {!ready || !result ? "Loading..." : JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  );
}
