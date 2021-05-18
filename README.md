# License Plate Recognition for Indian Scenarios

**The proposed approach is composed by four main steps:**
* ##### Image Upload
    Input image is taken from the user. 
    The image is then stored in a folder using a node js express server

* ##### Vehicle detection
    Given an input image, the first module detects vehicles in the scene.


* ##### License Plate localisation
    Within each detection region, the pretrained(WPOD-NET) searches for LPs and allowing a rectification of the LP area to a rectangle resembling a frontal view.

* ##### OCR
    These positive and rectified detections are fed to an OCR Network for final character recognition.

* ##### Displaying
    The cars are displayed on the frontend along with its license plate number

**Prerequisite Library for LPR.py:**
1) ##### Tensorflow
        pip install tensorflow==1.15.3
2) ##### Keras
        pip install keras==2.2.4
        
3) ##### Scikit-image
        pip install scikit-image
        
4) ##### Installing Tesseract 4 on Windows
    Follow the guide [here](https://medium.com/quantrium-tech/installing-and-using-tesseract-4-on-windows-10-4f7930313f82)


**Prerequisite Library for starting the server**:
1. Follow the guide [here](https://docs.microsoft.com/en-us/windows/dev-environment/javascript/nodejs-on-windows) for installing nvm-windows, node.js, and npm
2. ##### Installing nodemon 
        npm i nodemon


### Running the Project
##### Starting the backend
1. Open Windows Powershell/cmd on your system
2. Use command *cd "path/backend"* to navigate to the backend folder
3. Run command "npm install"
4. Run command *nodemon app.js*
5. The backend server is started at localhost:5000, if the port is already in use, change port number in app.js file
##### Starting the frontend
1. Open Windows Powershell/cmd on your system
2. Use command *cd "path/frontend"* to navigate to the frontend folder.
3. Run command "npx install"
4. Run command *npm start*
5. The frontend server is started at localhost:3000, if the port is already in use, edit *scripts/start. js* and find/replace 3000 with whatever port you want to use. 