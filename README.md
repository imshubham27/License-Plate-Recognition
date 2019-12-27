 # License Plate Recognition System

 ## About 

 Number Plate Detection with a Multi-Convolutional Neural Network Approach with Optical Character Recognition for Mobile Devices 
 I have tried to imporved the model what was proposed in the research paper [LINK](http://jips.jatsxml.org/Article/12/1/100)

 The license plate is detected by sequentially applying multiple CNN-verifiers. We first performed a supervised CNN for car detection (CNN1) and second, a supervised CNN for plate detection (CNN2). Plate validation was performed with OCR by detecting digits with a third CNN-verifier (CNN3). A sliding window was used to generate input images for all classifiers.

 Our work inculdes:
 1.**License plate candidates generation**
    The algorithm detects multiple LPs under differ￾ent image capture conditions and extracts them using edge
    statistics. Morphological operations are used to extract vertical edges of the LP regions while removing background.
 2.**CNN detection**
    Candidate generation technique described above detects all LP regions along with many non-license plate regions that are to be filtered out.
    We used CNN model for classifying candidate regions generated from previous stage into license plate or non-license plate regions.
 3.**LP Recognition**
 ⋅⋅* Performs text detection using OpenCV’s EAST text detector, a highly accurate deep learning text detector used to detect text in natural scene images.
 ⋅⋅* Once we have detected the text regions with OpenCV, we’ll then extract each of the text ROIs and pass them into Tesseract, enabling us to build an entire OpenCV        OCR pipeline!
    



 