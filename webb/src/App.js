import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';
import axios from 'axios';
import { Text, StyleSheet } from "react";
import { RotateCircleLoading } from 'react-loadingg';
import { setConfiguration } from 'react-grid-system';
import { Container, Row, Col } from 'react-grid-system';



function App() {
  const [images, setImages] = useState([])
  const [noFaceImages, setNoFaceImages] = useState([])
  const [prediction, setPrediction] = useState([])
  const [upload, setUpload] = useState("https://pbs.twimg.com/profile_images/740272510420258817/sd2e6kJy_400x400.jpg")
  const [isImage, setIsImage] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [showPredict, setShowPredict] = useState(false)

  const multipleFilesOnChange = (event) => {
    setImages(event.target.files)
    let bool = true
    for (var i = 0 ; i<event.target.files.length ; i++){
      let filename = event.target.files[i].name
      if (!filename.match(/.(jpg|jpeg|png|gif)$/i)) {
        setIsImage(false)
        setShowPredict(false)
        bool = false
        break
      }
    }

    if (bool) {
      setIsImage(true)
      setUpload(URL.createObjectURL(event.target.files[0]))
      setShowPredict(true)
    }
  }

  const sendMultipleImages = async () => {
    setNoFaceImages([])
    setPrediction([])

    let formData = new FormData()
    for (let i = 0; i < images.length; i++) {
      formData.append("files", images[i])
  }
    console.log(formData)

    fetch("http://localhost:4000/uploadMultipleFiles",{
      method:"post",
      body: formData,
    })
    .then((res) => res.text())
    .then((resBody) => {
      console.log('res body: ')
      console.log(resBody)
    })

    setIsLoading(true)
    let res =  await axios.get('http://localhost:4000/runPython')
    let wrong_filenames_str = res.data.split('*')[1];
    let wrong_filenames_arr = wrong_filenames_str.split(" ")
    setNoFaceImages(wrong_filenames_arr)

    let pred_data_str = res.data.split('*')[0]
    let pred_data_arr = (pred_data_str).split(" ")
    //the last index in split is garbage and need to be removed
    pred_data_arr.pop()
    const final_pred = pred_data_arr.reduce(function (pred, key, index) { 
    return (index % 3 == 0 ? pred.push([key]) 
      : pred[pred.length-1].push(key)) && pred;
  }, []);
    setPrediction(final_pred)
    setIsLoading(false)

  }


  let invalidImage = <div>
    <img src={"https://pbs.twimg.com/profile_images/740272510420258817/sd2e6kJy_400x400.jpg"} className="photo"/>
    <p className="warning">A file you chose is not an image, try again.</p>
    </div> 

  

return (  

  <div className="App">
    <header className="App-header">
    <Row >
      <Col><strong>Image</strong></Col>
      <Col><strong>Age</strong></Col>
      <Col><strong>Gender</strong></Col>
    </Row>
    {isImage ? 
            <div>
            <img src={upload} className="photo"/></div>
          :
          invalidImage
          }
    {isLoading ? 
        <RotateCircleLoading />
        :
        null
    }
  <div className="description">Number of images uploaded: {images.length}</div>
  <div className="description">Number of faces detected: {prediction.length}</div>
    
    <p/>

    <div className="description">Prediction:</div>
    {prediction.map((row, index) => ( 
  <Row >
    <Col>{row[0]}</Col>
    <Col>{row[1]}</Col>
    <Col>{row[2]}</Col>
  </Row>
))}

<p/>

  <div className="description">Faces could not be detected for:</div>
  {noFaceImages.map(row => ( 
  <Row >
  {row}
  </Row>
))}
  
  <p/>


    <input type="file" multiple name="file" className="input" onChange={multipleFilesOnChange} />
    {showPredict ? <button onClick={sendMultipleImages} className="button">Predict</button> : <div></div>}
    <a href='pred.csv' download='pred.csv'>Download spreadsheet file</a>

    </header>
  </div>
);

}

export default App;