const util = require("util");
const multer = require("multer");
const path = require('path');

//Store in Upload
const server_storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, 'Img'+path.extname(file.originalname));
    }
});

var uploadServer = multer({ storage: server_storage }).single("file");
var uploadFilesMiddlewareServer = util.promisify(uploadServer);

module.exports = {
    uploadFilesMiddlewareServer: uploadFilesMiddlewareServer
  };