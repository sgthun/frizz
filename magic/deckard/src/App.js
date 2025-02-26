import logo from './logo.svg';
import './App.css';
import WebcamCapture from './WebcamCapture';
import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

function App() {
  const [model, setModel] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [ticker, setTicker] = useState(0);
  const [startTime, setStartTime] = useState(Date.now());
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  
  const classNames = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika',
    'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach',
    'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
  ];

  // Load the model once when the component mounts.
  useEffect(() => {
    tf.loadLayersModel(process.env.PUBLIC_URL + '/food_model_50/model.json')
      .then(loadedModel => {
        setModel(loadedModel);
        console.log("Model loaded successfully");
        setStartTime(Date.now());
      })
      .catch(err => console.error("Error loading model:", err));
  }, []);

  // Set up a timer to update elapsed seconds.
  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [startTime]);

  // Run inference whenever capturedImage changes and the model is loaded.
  useEffect(() => {
    if (capturedImage && model) {
      // Create an image element to load the data URL.
      const img = new Image();
      img.src = capturedImage;
      img.onload = () => {
        // Convert the image to a tensor. Adjust target size as needed.
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([180, 180])
          .expandDims(0).toFloat();
          
        // Run inference on the tensor.
        const predictionTensor = model.predict(tensor);
        // Compute the softmax probabilities.
        const softmaxPrediction = tf.softmax(predictionTensor);
        // Get the index with the maximum probability.
        const predictedIndex = tf.argMax(softmaxPrediction, 1);
        const scoresArray = softmaxPrediction.dataSync();
        let scoreMap = {};
        for (let i = 0; i < softmaxPrediction.shape[1]; i++) {
          scoreMap[classNames[i]] = (scoresArray[i] * 100).toFixed(2) + '%';
          console.log(
            `Class ${classNames[i]}: ${(scoresArray[i] * 100).toFixed(2)}%`
          );
        }
        setPrediction(scoreMap);
        
        console.log("Predicted class:", classNames[predictedIndex.dataSync()[0]]);
        
        // Increment ticker.
        setTicker(prevTicker => prevTicker + 1);
        
        // Clean up tensors to free memory.
        tensor.dispose();
        predictionTensor.dispose();
        softmaxPrediction.dispose();
        predictedIndex.dispose();
      };
    }
  }, [capturedImage, model]);

  // Calculate images per second.
  const imagesPerSecond = elapsedSeconds > 0 ? (ticker / elapsedSeconds).toFixed(2) : 0;

   // Calculate the maximum score as a number from the prediction map.
  const maxScore = prediction
    ? Math.max(...Object.values(prediction).map(score => parseFloat(score)))
    : 0;
  
  return (
    <div className="App">
      <WebcamCapture 
        ticker={ticker} 
        capturedImage={capturedImage} 
        setCapturedImage={setCapturedImage}
      />
      <p>Model: {model ? "Loaded" : "Not loaded"}</p>
      <p>Ticks: {ticker}</p>
      <p>Elapsed seconds: {elapsedSeconds}</p>
      <p>Images per second: {imagesPerSecond}</p>
      
      {prediction && (
        <table border="1" cellPadding="5" style={{ margin: '20px auto', textAlign: 'center' }}>
          <thead>
            <tr>
              <th>Class</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(prediction).map(([className, score]) => {
              // Convert score string to a number.
              const numericScore = parseFloat(score);
              const rowStyle = numericScore === maxScore ? { backgroundColor: 'lightgreen' } : {};
              return (
                <tr key={className} style={rowStyle}>
                  <td>{className}</td>
                  <td>{score}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default App;