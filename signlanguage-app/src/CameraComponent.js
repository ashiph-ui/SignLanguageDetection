// Import necessary React hooks and the react-webcam component
import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';

// Webcam video settings (224x224, front-facing camera)
const videoConstraints = {
  width: 224,
  height: 224,
  facingMode: 'user',
};

function CameraComponent() {
  // Reference to the webcam component
  const webcamRef = useRef(null);

  // Store the predicted label returned from the backend
  const [prediction, setPrediction] = useState('');

  // useEffect: runs once on mount, sets up frame capture every 3 seconds
  useEffect(() => {
    const interval = setInterval(async () => {
      // Ensure the webcam is available
      if (webcamRef.current) {
        // Capture a frame from the webcam
        const imageSrc = webcamRef.current.getScreenshot();

        if (imageSrc) {
          try {
            // TODO: Replace with your actual backend URL when known
            const response = await fetch('http://localhost:8000/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ image: imageSrc }),
            });

            // Parse and store the prediction from backend
            const result = await response.json();
            setPrediction(result.label);
          } catch (err) {
            console.error('Error sending image to backend:', err);
          }
        }
      }
    }, 3000); // Capture a frame every 3 seconds

    // Clean up the interval on unmount
    return () => clearInterval(interval);
  }, []);

  // Render the webcam and the current prediction
  return (
    <div className="camera-container">
      <Webcam
        audio={false}
        ref={webcamRef}
        height={224}
        width={224}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
      />
      <p>Prediction: {prediction}</p>
    </div>
  );
}

export default CameraComponent;

