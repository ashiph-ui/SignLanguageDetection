// Import necessary React hooks and the react-webcam component
import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';

// Define video constraints for the webcam feed
const videoConstraints = {
  width: 224,
  height: 224,
  facingMode: 'user', // Use the front-facing camera
};

function CameraComponent() {
  // Create a reference to the Webcam component
  const webcamRef = useRef(null);

  // State to store the prediction result from the backend
  const [prediction, setPrediction] = useState('');

  // useEffect will run once when the component mounts
  // Sets up an interval to capture an image every 3 seconds and get a prediction
  useEffect(() => {
    const interval = setInterval(async () => {
      // Make sure the webcam is ready and can capture an image
      if (webcamRef.current) {
        // Capture the current frame from the webcam
        const imageSrc = webcamRef.current.getScreenshot();

        // If a valid image was captured, send it to the backend
        if (imageSrc) {
          try {
            const res = await fetch('http://localhost:8000/predict', {
              method: 'POST',
              body: JSON.stringify({ image: imageSrc }),
              headers: { 'Content-Type': 'application/json' },
            });

            // Parse the prediction response from the backend
            const result = await res.json();

            // Update the prediction state
            setPrediction(result.label);
          } catch (error) {
            console.error('Prediction fetch error:', error);
          }
        }
      }
    }, 3000); // Capture an image every 3 seconds

    // Clear the interval when the component unmounts
    return () => clearInterval(interval);
  }, []);

  // Return the webcam view and the current prediction result
  return (
    <div className="camera-container">
      <Webcam
        audio={false}
        height={224}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={224}
        videoConstraints={videoConstraints}
      />
      {/* Display the latest prediction */}
      <p>Prediction: {prediction}</p>
    </div>
  );
}

export default CameraComponent;
