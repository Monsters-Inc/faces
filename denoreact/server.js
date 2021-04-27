import { Application } from "https://deno.land/x/abc@v1.3.1/mod.ts";
//import App from "./src/App";

const app = new Application();

app
  .get("/app", (c) => {
    return "console_logs";
  })
  .static("/", "build")
  .file('/', 'build/index.html')
  .start({ port: 8080 });

//   const cmd = Deno.run({
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