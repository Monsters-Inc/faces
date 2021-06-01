import React, { useState } from 'react';
import './App.css';
import axios from 'axios';
import { RotateCircleLoading } from 'react-loadingg';
import { Row, Col } from 'react-grid-system';
import { CSVLink } from "react-csv";

function App() {
  const [images, setImages] = useState([])
  const [wrongFaceImages, setWrongFaceImages] = useState([])
  const [prediction, setPrediction] = useState([])
  const [upload, setUpload] = useState("https://pbs.twimg.com/profile_images/740272510420258817/sd2e6kJy_400x400.jpg")
  const [isImage, setIsImage] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [showPredict, setShowPredict] = useState(false)
  const [isDone, setIsDone] = useState(false)

  const multipleFilesOnChange = (event) => {
    setIsDone(false)
    setWrongFaceImages([])
    setPrediction([])
    setImages(event.target.files)
    let bool = true
    for (var i = 0; i < event.target.files.length; i++) {
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
    let formData = new FormData()
    for (let i = 0; i < images.length; i++) {
      formData.append("files", images[i])
    }

    fetch("http://localhost:4000/uploadMultipleFiles", {
      method: "post",
      body: formData,
    })
      .then((res) => res.text())
      .then((resBody) => {
        console.log('res body: ')
        console.log(resBody)
      })

    setIsLoading(true)
    let res = await axios.get('http://localhost:4000/runPython')
    let wrong_filenames_str = res.data.split('*')[1];
    let wrong_filenames_arr = wrong_filenames_str.split(" ")
    wrong_filenames_arr.shift()
    wrong_filenames_arr.pop()
    setWrongFaceImages(wrong_filenames_arr)

    let pred_data_str = res.data.split('*')[0]
    let pred_data_arr = (pred_data_str).split(" ")
    //the last index in split is garbage and need to be removed
    pred_data_arr.pop()
    const final_pred = pred_data_arr.reduce(function (pred, key, index) {
      return (index % 3 === 0 ? pred.push([key])
        : pred[pred.length - 1].push(key)) && pred;
    }, []);
    setPrediction(final_pred)
    setIsLoading(false)
    setIsDone(true)
    setShowPredict(false)
  }

  const restart = async () => {
    setIsDone(false)
    setWrongFaceImages([])
    setPrediction([])
    setImages([])
    setUpload("https://pbs.twimg.com/profile_images/740272510420258817/sd2e6kJy_400x400.jpg")
  }

  const cropFileName = filename => {
    if (filename.length > 10) {
      filename = filename.slice(0, 6) + "..."
      return filename
    }
    else { 
      return filename
    }
  }

  let invalidRows = []
  wrongFaceImages.forEach(row => {
    invalidRows.push([row, '-', '-'])
  })

  const csvData = [
    ["Image", "Age", "Gender"],
    ...prediction,
    ...invalidRows
  ];


  let invalidImage = <>
    <img src={"https://pbs.twimg.com/profile_images/740272510420258817/sd2e6kJy_400x400.jpg"} alt="Invalid" className="photo" />
    <p className="warning">A file you chose is not an image, try again.</p>
  </>

  return (

    <div className="App">
      <header className="App-header">

        {isImage ?
          <div>
            <img src={upload} className="photo" alt="Uploaded" /></div>
          :
          invalidImage
        }
        {isLoading ?
          <RotateCircleLoading />
          :
          null
        }
        {isDone && <div className="description">Number of images uploaded: {images.length}</div>}
        {isDone && <div className="description">Number of faces detected: {prediction.length}</div>}

        {!isDone && <Row className='row'>
          <Col xs={7}><label className="custom-file-upload">
            <input type="file" multiple name="file" className="input" onChange={multipleFilesOnChange} />
        Upload Images
    </label>
          </Col>
          <Col>
            <div className="fileChecker">{images.length > 0 ? images.length : 'No'} file{images.length > 1 ? 's' : ''} uploaded</div>
          </Col>
        </Row>
        }
        {isDone && <button onClick={restart} className="button">Restart</button>}
        {!isDone && <div className="info">
          This application predicts the gender and age of people in, primarily, historical photographs. Please upload
          one or multiple images for analysis.
    <br />
          <br />
    Good luck!
  </div>
        }

        {isDone && <CSVLink className="csv_link" filename="predictions.csv" data={csvData} separator={";"}><button className="csvButton">Download Results (.csv)</button></CSVLink>}

        {isDone &&
          <div className="predictionWrapper">
            <div className="prediction">Prediction:</div>
            <Row>
              <Col><strong>Image</strong></Col>
              <Col><strong>Age</strong></Col>
              <Col><strong>Gender</strong></Col>
            </Row>
        {prediction.map((row, index) => (
          <Row key={index}>
            <Col>{cropFileName(row[0])}</Col>
            <Col>{row[1]}</Col>
            <Col>{row[2]}</Col>
          </Row>
        ))}
        </div>
        }

        {isDone && wrongFaceImages.length !== 0 && <div className="notDetected"><strong className="headerNotDetected" >Faces could not be detected for:</strong>
        {wrongFaceImages.map((row, index) => 
          <Row key={index}>
            <Col xs={12}>{row}</Col>
          </Row>
        )}
        </div>}

        {showPredict ? <button onClick={sendMultipleImages} className="button">Predict</button> : <div></div>}

      </header>
    </div>
  );

}

export default App;