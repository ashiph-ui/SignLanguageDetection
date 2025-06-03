import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';

const videoConstraints = {
  width: 224,
  height: 224,
  facingMode: 'user',
};

function CameraComponent() {
  const webcamRef = useRef(null);
  const [prediction, setPrediction] = useState('');

  useEffect(() => {
    const interval = setInterval(async () => {
      if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();

        if (imageSrc) {
          // Convert base64 to blob
          const blob = await fetch(imageSrc).then(res => res.blob());

          // Create FormData with the image blob as a file
          const formData = new FormData();
          formData.append('file', blob, 'capture.jpg');

          try {
            const response = await fetch('http://localhost:8000/predict/', {
              method: 'POST',
              body: formData,
            });

            const result = await response.json();
            setPrediction(result.prediction); // match key with backend response
          } catch (err) {
            console.error('Error sending image to backend:', err);
          }
        }
      }
    }, 3000);

    return () => clearInterval(interval);
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

