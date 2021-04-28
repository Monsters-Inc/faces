const express = require('express')
var cors = require('cors')
const app = express()

app.use(cors()) 
app.get('/', (req, res) => {

    
    const { spawn } = require('child_process');
    const pyProg = spawn('python3', ['../main.py', req.query[0]]);

    pyProg.stderr.on('data', function(data) {
        console.log('ERROR')
        console.log(data.toString())
    })

    pyProg.stdout.on('data', function(data) {

        console.log(data.toString());
        res.write(data);
        res.end()
    });
})

app.listen(4000, () => console.log('Application listening on port 4000!'))