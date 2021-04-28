import logo from './logo.svg';
import React, { useState } from 'react';
import './App.css';
import axios from 'axios';
import { Text, StyleSheet } from "react";

//import { Application } from "https://deno.land/x/abc@v1.3.1/mod.ts";
// import test from './server.js';

function App() {

const [file, setFile] = useState(0);

const test = async () => {
  let res =  await axios.get('http://localhost:4000', {params: file})
}

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>Enter the image file name:</p>
        <input type="text" placeholder="Filename" onChange={e => setFile(e.target.value)} />
        <button onClick={test}>Predict!</button>
      </header>
    </div>
  );
}

export default App;
