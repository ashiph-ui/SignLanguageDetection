import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';

// Webcam settings
const videoConstraints = {
  width: 224,
  height: 224,
  facingMode: 'user',
};

function CameraComponent() {
  const webcamRef = useRef(null); // Webcam reference
  const [prediction, setPrediction] = useState(''); // Prediction state

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
          } catch (err) {
            console.error('Error sending image to backend:', err);
          }
        }
      }
    }, 3000); // Capture and send every 3 seconds

    return () => clearInterval(interval); // Clean up the interval on unmount
  }, []);

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

