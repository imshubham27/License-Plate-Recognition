const express = require("express");
const router = express.Router();
const path = require('path');
const fs = require('fs');

const {spawn} = require('child_process');

const upload = require("../middleware/upload");

const uploadFile = async (req, res) => {
  try {
    await upload.uploadFilesMiddlewareServer(req, res);

    console.log(req.file);
    if (req.file == undefined) {
      return res.send(`You must select a file.`);
    }
    const image_path=req.file.path;
    return res.send('File uploaded');
  } catch (error) {
    console.log(error);
    return res.send(`Error when trying upload image: ${error}`);
  }
};

const img = async (req, res) => {
  const uploadDirectory = path.join('images');
  fs.readdir(uploadDirectory, (err, files)=>{
    if(err){
      return res.json({msg:err})
    }
    if(files.length===0){
      return res.json({msg:'No Image Uploaded'});
    }

    return res.json({files})
  })
};


const py =async (req, res) => {
 
  var dataToSend;
  // spawn new child process to call the python script
  const python = spawn('python', ['./LPR.py']);
  // collect data from script
  python.stdout.on('data',  (data) => {
   console.log('Pipe data from python script ...');
   dataToSend = data.toString();
   console.log(dataToSend);
  });

  // in close event we are sure that stream from child process is closed
  python.on('close', (code) => {
  console.log(`child process close all stdio with code ${code}`);
  // send data to browser
  res.send(dataToSend)
  });

  return dataToSend;
};


router.post("/upload", uploadFile);

router.get("/py",py)

router.get("/images",img)

module.exports = router;