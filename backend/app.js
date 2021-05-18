const express = require("express");
const cors = require('cors');

const app = express();
app.use(cors());

//ROUTES
app.use('/', require('./routes/route'));

app.use(express.static('images'));  


const PORT = process.env.PORT || 5000;
app.listen(PORT, console.log(`Server started at port ${PORT}`));
