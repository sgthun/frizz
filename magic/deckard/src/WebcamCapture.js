import React, { useRef, useState, useEffect, use } from 'react';

function WebcamCapture({capturedImage, setCapturedImage, ticker}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    // Ask for access to the webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(err => {
        console.error("Error accessing webcam: ", err);
      });
  }, []);
  useEffect(() => {
    handleCapture();
  }, [ticker]);
  const handleCapture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      // Set the canvas size to the video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      // Draw the current video frame to the canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      // Convert the canvas to a data URL and save it to state
      const imageData = canvas.toDataURL('image/png');
      setCapturedImage(imageData);
    }
  };

  return (
    <div>
      <h2>Webcam Capture</h2>
      <video ref={videoRef} autoPlay playsInline style={{ width: '100%', maxWidth: '400px' }} />
      <br />
      <button onClick={handleCapture}>Capture Photo</button>
      {/* Hidden canvas used for taking snapshot */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      
      {capturedImage && (
        <div>
          <h3>Captured Image:</h3>
          <img src={capturedImage} alt="Captured" style={{ width: '100%', maxWidth: '400px' }} />
        </div>
      )}
    </div>
  );
}

export default WebcamCapture;