import logo from './logo.svg';
import React, { useState } from 'react';
import './App.css';
import axios from 'axios';
import { Text, StyleSheet } from "react";

//import { Application } from "https://deno.land/x/abc@v1.3.1/mod.ts";
// import test from './server.js';

function App() {

const [file, setFile] = useState(0);
const [answer, setAnswer] = useState("");
const [image, setImage] = useState(null)

const test = async () => {
  let res =  await axios.get('http://localhost:4000', {params: file})
  console.log(res.data)
  var gender = res.data
  setAnswer(gender)
}

const fileHandler = (event) => {
  console.log(event.target.files[0])
  setImage(event.target.files[0])
}

const fileUploadHandler = (event) => {

}

  return (
    <div className="App">
      <header className="App-header">
      <input type="file" onChange={fileHandler}/>
      <button onClick={fileUploadHandler}>Upload</button>

        {/* <img src={logo} className="App-logo" alt="logo" />
        <p>Enter the image file name:</p>
        <input type="text" placeholder="Filename" onChange={e => setFile(e.target.value)} />
        <button onClick={test}>Predict!</button>
        <p>Gender: {answer}</p> */}

      </header>
    </div>
  );
}

export default App;
