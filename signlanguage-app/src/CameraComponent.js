import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';

// Webcam settings — higher resolution for a crisp feed; the backend extracts
// MediaPipe landmarks, so it is resolution-independent.
const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: 'user',
};

// Minimum gap between the end of one prediction request and the start of the
// next. The loop is self-scheduling, so this is a floor, not a fixed cadence.
const POLL_INTERVAL_MS = 500;

// How many *consecutive* connection-level failures before we declare the
// backend offline. A single dropped/slow frame then self-heals silently.
const OFFLINE_THRESHOLD = 3;

function CameraComponent() {
  const webcamRef = useRef(null); // Webcam reference
  const [prediction, setPrediction] = useState(''); // Prediction state
  const [confidence, setConfidence] = useState(null); // Model confidence (0–1)
  const [backendUp, setBackendUp] = useState(true); // Backend reachability

  useEffect(() => {
    let cancelled = false; // Set on unmount to stop the loop
    let timerId = null; // Pending setTimeout handle for cleanup
    let failures = 0; // Consecutive connection-level failures

    // Send a single frame to the backend. Returns nothing; updates state.
    const sendFrame = async () => {
      if (!webcamRef.current) return;

      // Capture the current frame as a base64 image
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;

      try {
        // Convert base64 to Blob
        const blob = await fetch(imageSrc).then(res => res.blob());

        // Prepare the form data with the image file
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');

        // Send the image to the backend
        const response = await fetch('http://localhost:8000/predict/', {
          method: 'POST',
          body: formData,
        });
        if (cancelled) return;

        if (!response.ok) {
          // The server answered, so it is reachable — this is a per-request
          // problem (e.g. 500 on an undecodable frame). Reset the offline
          // counter and keep the last good prediction rather than blanking.
          failures = 0;
          setBackendUp(true);
          return;
        }

        // Parse the backend response
        const result = await response.json();
        if (cancelled) return;

        failures = 0;
        setPrediction(result.prediction); // Update prediction in UI
        setConfidence(result.confidence ?? null);
        setBackendUp(true);
      } catch (err) {
        if (cancelled) return;
        // Connection-level failure (fetch rejected / JSON parse failed).
        // Only declare the backend offline after several in a row; keep the
        // last prediction on screen so a single dropped frame is invisible.
        console.error('Error sending image to backend:', err);
        failures += 1;
        if (failures >= OFFLINE_THRESHOLD) {
          setBackendUp(false);
        }
      }
    };

    // Self-scheduling loop: only ever one request in flight, and the next is
    // scheduled after the previous completes — so a slow frame can never stack.
    const loop = async () => {
      await sendFrame();
      if (!cancelled) {
        timerId = setTimeout(loop, POLL_INTERVAL_MS);
      }
    };

    loop();

    return () => {
      cancelled = true;
      if (timerId) clearTimeout(timerId);
    };
  }, []);

  // Decide what to show in the prediction slot
  const isLetter = prediction && prediction.length === 1;
  let display, hint;
  if (!backendUp) {
    display = '·';
    hint = 'Backend offline — start the FastAPI server on port 8000';
  } else if (isLetter) {
    display = prediction;
    hint = 'Detected letter';
  } else if (prediction === 'Blank') {
    display = '—';
    hint = 'Blank sign detected';
  } else if (prediction === 'No hand detected') {
    display = '·';
    hint = 'No hand detected — raise your hand into frame';
  } else {
    display = '·';
    hint = 'Waiting for first prediction…';
  }

  return (
    <div className="camera-container">
      <div className="webcam-frame">
        <span className="live-badge">
          <span className={backendUp ? 'live-dot' : 'live-dot offline'} />
          {backendUp ? 'LIVE' : 'OFFLINE'}
        </span>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          mirrored
        />
      </div>

      <div className="prediction-panel">
        <span className="prediction-label">Prediction</span>
        <span className={isLetter ? 'prediction-value' : 'prediction-value idle'}>
          {display}
        </span>
        {backendUp && confidence != null && (
          <div className="confidence">
            <div className="confidence-track">
              <div
                className={confidence >= 0.75 ? 'confidence-fill high' : 'confidence-fill'}
                style={{ width: `${Math.round(confidence * 100)}%` }}
              />
            </div>
            <span className="confidence-pct">{Math.round(confidence * 100)}% confident</span>
          </div>
        )}
        <span className="prediction-hint">{hint}</span>
      </div>
    </div>
  );
}

export default CameraComponent;
