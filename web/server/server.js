const express = require("express")
const app = express()
const multer = require("multer")
const cors = require("cors")
const upload = multer({dest: "./uploads"})
const fs = require("fs")
const PORT = 4000
const path = require('path');

const uploadsExists = fs.existsSync('./uploads');
const preprocessedUploadsExists = fs.existsSync('./preprocessedUploads');
const uploadsWithoutFaceExists = fs.existsSync('./uploadsWithoutFace');

if(!uploadsExists){
    fs.mkdirSync('./uploads')
}

if(!uploadsWithoutFaceExists){
    fs.mkdirSync('./uploadsWithoutFace')
}

if(!preprocessedUploadsExists){
    fs.mkdirSync('./preprocessedUploads')
}

app.use(cors())
app.use("/static", express.static("./uploads"))
app.post("/uploadMultipleFiles", upload.array("files"), (req,res) => {
    for (i in req.files) {
        file = req.files[i]
        let newFileName = file.originalname//file.filename + "." + fileType
        console.log(`./uploads/${newFileName}`)
        fs.rename(
        `./uploads/${file.filename}`,
        `./uploads/${newFileName}`,
        function () {
            console.log("callback")
            //res.send("200")
        }
    )
    }
})

app.get("/runPython", (req,res) => {
    const { spawn } = require('child_process');
    const pyProg = spawn('python3', ['../../main.py'])//, req.query[0]]);

    pyProg.stderr.on('data', function(data) {
        console.log(data.toString())
    })

    pyProg.stdout.on('data', function(data) {
        console.log('RESPONSEN: ')
        console.log(data.toString());
        res.write(data);
        res.end()
    })
})

app.listen(PORT, () => console.log(`App listening on port ${PORT}`))