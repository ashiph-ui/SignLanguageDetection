import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';

// Webcam settings — higher resolution for a crisp feed; the backend extracts
// MediaPipe landmarks, so it is resolution-independent.
const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: 'user',
};

function CameraComponent() {
  const webcamRef = useRef(null); // Webcam reference
  const [prediction, setPrediction] = useState(''); // Prediction state
  const [confidence, setConfidence] = useState(null); // Model confidence (0–1)
  const [backendUp, setBackendUp] = useState(true); // Backend reachability

  useEffect(() => {
    const interval = setInterval(async () => {
      if (webcamRef.current) {
        // Capture the current frame as a base64 image
        const imageSrc = webcamRef.current.getScreenshot();

        if (imageSrc) {
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

            // Parse the backend response
            const result = await response.json();
            setPrediction(result.prediction); // Update prediction in UI
            setConfidence(result.confidence ?? null);
            setBackendUp(true);
          } catch (err) {
            console.error('Error sending image to backend:', err);
            setBackendUp(false);
          }
        }
      }
    }, 3000); // Capture and send every 3 seconds

    return () => clearInterval(interval); // Clean up the interval on unmount
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
