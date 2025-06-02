// Import necessary React hooks and the react-webcam component
import React, { useRef, useCallback, useState } from 'react';
import Webcam from 'react-webcam';

// Set constraints for the webcam video
const videoConstraints = {
  width: 224,
  height: 224,
  facingMode: 'user', // Use front-facing camera
};

function CameraComponent() {
  // Create a reference to the webcam component so we can access it directly
  const webcamRef = useRef(null);

  // State to store the predicted label from the server
  const [prediction, setPrediction] = useState('');

  // Function to capture the current image from the webcam
  const capture = useCallback(async () => {
    // Take a screenshot from the webcam feed
    const imageSrc = webcamRef.current.getScreenshot();

    // Send the image to the FastAPI backend for prediction
    const res = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: JSON.stringify({ image: imageSrc }),
      headers: { 'Content-Type': 'application/json' },
    });

    // Get the prediction result from the server
    const result = await res.json();

    // Update the prediction in the UI
    setPrediction(result.label);
  }, [webcamRef]);

  // Return the JSX to display the webcam and prediction result
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
    <button onClick={capture}>Predict</button>
    <p>Prediction: {prediction}</p>
  </div>
);


}

export default CameraComponent;
