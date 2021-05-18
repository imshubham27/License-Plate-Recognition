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

#### Indian License Plate Dataset along with annotation: [Link](https://drive.google.com/drive/folders/1XEzy56gdV0DrVwvIvYNdcLJbLIn4ehpm?usp=sharing)

## Project Execution
1. ##### Clone the repository on your local machine 

2. ##### Before running the projects, Some files need to be downloaded into Backend folder:
    * Pretrained [Yolov3 model 416X416](https://drive.google.com/file/d/1iWnW7-95zs1jUM8X6W5xBrlHLwKcA0YT/view?usp=sharing)
    * Frozen [East Text Detector](https://drive.google.com/file/d/1wNrHp3pAXcQWWAfsAtRNVZ8AsTsHnQbb/view?usp=sharing)
    * [Yolov3 416X416 weights](https://drive.google.com/file/d/1U1nKGd2mcSL3isNqyQbRlhBcRh748EOq/view?usp=sharing) (incase you want to train your own custom object detection model)

3. ##### Starting the backend
    * Open Windows Powershell/cmd on your system
    * Use command *cd "path/backend"* to navigate to the backend folder
    * Run command "npm install"
    * Run command *nodemon app.js*
    * The backend server is started at localhost:5000, if the port is already in use, change port number in app.js file
4. ##### Starting the frontend
    * Open Windows Powershell/cmd on your system
    * Use command *cd "path/frontend"* to navigate to the frontend folder.
    * Run command "npx install"
    * Run command *npm start*
    * The frontend server is started at localhost:3000, if the port is already in use, edit *scripts/start. js* and find/replace 3000 with whatever port you want to use. 