import logo from './logo.svg';
import './App.css';
import { Application } from "https://deno.land/x/abc@v1.3.1/mod.ts";


function App() {

// const denoPython = async () => { const cmd = Deno.run({
//     cmd: ["python3", "../main.py"], 
//     stdout: "piped",
//     stderr: "piped"
//   });
  
//   const output = await cmd.output() // "piped" must be set
//   const outStr = new TextDecoder().decode(output);
  
//   const error = await cmd.stderrOutput();
//   const errorStr = new TextDecoder().decode(error);
  
//   cmd.close(); // Don't forget to close it
  
//   console.log(outStr, errorStr);
// }

const test = () => {
  console.log('FUNKADE')
}

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <button onClick={test}>Click me!</button>
        <p>HALLÖJSNA</p>
      </header>
    </div>
  );
}

export default App;
