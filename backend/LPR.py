import tensorflow as tf
import keras
import cv2
import time
import numpy as np
import os
from os.path import splitext,basename
from keras.models import model_from_json
import glob
import cv2# For OpenCV modules (For Image I/O and Contour Finding)
import numpy as np
import re
import scipy
import math
from skimage import img_as_ubyte
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import pytesseract
import shutil
import random
try:
 from PIL import Image
except ImportError:
 import Image
from imutils.object_detection import non_max_suppression
from pytesseract import Output
import scipy.fftpack # For FFT2
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from skimage.io import imsave
import matplotlib.pyplot as plt
def load_model1(path):
    try:
        path = splitext(path)[0]
        with open('wpod-net.json', 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('wpod-net.h5')
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "wpod-net.json"
wpod_net = load_model1(wpod_net_path)

# load yolov3 model
model = load_model('model.h5')

net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
from local_utils import detect_lp


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh):
                continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap(
        [box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap(
        [box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0:
                continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

# load and prepare an image


def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

# get all of the results above a threshold


def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores

# draw all results


def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        crop_img = data[y1:y1+height, x1:x1+width]
        # plt.imshow(crop_img)
        if v_labels[i] == "car" or v_labels[i] == "motorbike" or v_labels[i] == "truck" or v_labels[i] == "bus":
            plt.imsave("CARs\CR_"+f"{i+1}"+".png", crop_img)

        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')


# define the expected input shape for the model
input_w, input_h = 416, 416
# define our new photo
photo_filename = r'F:\Study\Final_Year_Project\backend\uploads\Img.jpg'
# photo_filename = r'F:\Study\Final_Year_Project\backend\Inputs\up.jpg'
# load and prepare image
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
# make prediction
yhat = model.predict(image)
# summarize the shape of the list of arrays
print([a.shape for a in yhat])
# define the anchors
anchors = [[116, 90, 156, 198, 373, 326], [
    30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
# define the probability threshold for detected objects
class_threshold = 0.9
boxes = list()
for i in range(len(yhat)):
    # decode the output of the network
    boxes += decode_netout(yhat[i][0], anchors[i],
                           class_threshold, input_h, input_w)
# correct the sizes of the bounding boxes for the shape of the image
correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
# suppress non-maximal boxes
do_nms(boxes, 0.5)
# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# get the details of the detected objects
v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
# summarize what we found
for i in range(len(v_boxes)):
    print(v_labels[i], v_scores[i])
# draw what we found
# print("HERE")
draw_boxes(photo_filename, v_boxes, v_labels, v_scores)


fileList1 = glob.glob(r'F:\Study\Final_Year_Project\backend\CARs\*.png')
if len(fileList1) == 0:
    a = cv2.imread(photo_filename)
    plt.imsave(r"F:\Study\Final_Year_Project\backend\CARs\CR_0.png", a)


def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


# Create a list of image paths
image_paths = glob.glob(r"F:\Study\Final_Year_Project\backend\CARs\*.png")
print("Found %i images..." % (len(image_paths)))

# forward image through model and return plate's image and coordinates
# if error "No Licensese plate is founded!" pop up, try to adjust Dmin

CORD =[]
def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, _, cor = detect_lp(
        wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor


# Obtain plate image and its coordinates from an image
for i in range(len(image_paths)):
    test_image = image_paths[i]
    print(test_image)
    try:
        LpImg, cor = get_plate(test_image)
        print("LEN", len(LpImg))
        print("Detect %i plate(s) in" %
              len(LpImg), splitext(basename(test_image))[0])
        CORD.append(cor)
    # print("Coordinate of plate(s) in image: \n", cor)
        # Visualize our result
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.axis(False)
        # plt.imshow(preprocess_image(test_image))
        # plt.subplot(1,2,1)
        for j in range(len(LpImg)):
            plt.axis(False)
            plt.imshow(LpImg[j])
            plt.imsave(r"F:\Study\Final_Year_Project\backend\License Plates\LP_" +
                       f"{i}"+f"{j}"+".png", LpImg[j])
    except AssertionError:
        os.remove(image_paths[i])
        continue


class ifChar:
    # this function contains some operations used by various function in the code
    def __init__(self, cntr):
        self.contour = cntr

        self.boundingRect = cv2.boundingRect(self.contour)

        [x, y, w, h] = self.boundingRect

        self.boundingRectX = x
        self.boundingRectY = y
        self.boundingRectWidth = w
        self.boundingRectHeight = h

        self.boundingRectArea = self.boundingRectWidth * self.boundingRectHeight

        self.centerX = (self.boundingRectX +
                        self.boundingRectX + self.boundingRectWidth) / 2
        self.centerY = (self.boundingRectY +
                        self.boundingRectY + self.boundingRectHeight) / 2

        self.diagonalSize = math.sqrt(
            (self.boundingRectWidth ** 2) + (self.boundingRectHeight ** 2))

        self.aspectRatio = float(self.boundingRectWidth) / \
            float(self.boundingRectHeight)


class PossiblePlate:

    def __init__(self):
        self.Plate = None
        self.Grayscale = None
        self.Thresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""


# this function is a 'first pass' that does a rough check on a contour to see if it could be a char
def checkIfChar(possibleChar):
    if (possibleChar.boundingRectArea > 80 and possibleChar.boundingRectWidth > 2
            and possibleChar.boundingRectHeight > 8 and 0.25 < possibleChar.aspectRatio < 1.0):

        return True
    else:
        return False


# check the center distance between characters
def distanceBetweenChars(firstChar, secondChar):
    x = abs(firstChar.centerX - secondChar.centerX)
    y = abs(firstChar.centerY - secondChar.centerY)

    return math.sqrt((x ** 2) + (y ** 2))


# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
    # sides to calculate the angle
    adjacent = float(abs(firstChar.centerX - secondChar.centerX))
    opposite = float(abs(firstChar.centerY - secondChar.centerY))

    # check to make sure we do not divide by zero if the center X positions are equal
    # float division by zero will cause a crash in Python
    if adjacent != 0.0:
        angleInRad = math.atan(opposite / adjacent)
    else:
        angleInRad = 1.5708

    # calculate angle in degrees
    angleInDeg = angleInRad * (180.0 / math.pi)

    return angleInDeg


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.7:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


for filename in os.listdir(r'F:\Study\Final_Year_Project\backend\License Plates'):
    print(filename)
    im = cv2.imread(r"F:\Study\Final_Year_Project\backend\License Plates\{}".format(filename))
    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = 20
    border = cv2.copyMakeBorder(
        im,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )

    plt.imshow(im)
    # plt.imshow(bottom)
    plt.imshow(border)
    cv2.imwrite(r'F:\Study\Final_Year_Project\backend\License Plates\{}'.format(filename), border)

    # Main program
s=[]
for filename in os.listdir(r'F:\Study\Final_Year_Project\backend\License Plates'):
    img = cv2.imread(r'F:\Study\Final_Year_Project\backend\License Plates\{}'.format(filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # img = cv2.bilateralFilter(img,9,75,75)
    # img = cv2.medianBlur(img, 5)
    # img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((1,1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                       cv2.THRESH_BINARY, 199, 5)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +
                                 cv2.THRESH_OTSU)
    # thresh1 = cv2.bitwise_not(thresh1)
    plt.imshow(img)
    plt.imshow(thresh1)
    # rotated = cv2.imread('/content/drive/MyDrive/8th/EXECUTION/License Plates/'+filename,0)
    # image = cv2.merge((img,img,img))
    image = cv2.merge((thresh1, thresh1, thresh1))
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (320, 320)
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * 3)
        dY = int((endY - startY) * 3)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 1, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        # '--oem 1 -l eng --psm 7 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        # "-l eng --oem 1 --psm 7 "
        config1 = ('--oem 1 -l eng --psm 7')
        text = pytesseract.image_to_string(roi, config=config1)
        # print("TEXT",text)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r: r[0][1])
    print(results)
    if len(results) == 0:
        s.append("UNABLE TO RECOGNIZE")
        continue
    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("OCR TEXT")
        print("========")
        # print("{}\n".format(text))
        print(re.sub(r'\W+', '', text))
        s.append(str((re.sub(r'\W+', '', text))))

        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # show the output image
        plt.imshow(output)

counter=0
def draw_box(image_path, cor, thickness=3): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    vehicle_image = preprocess_image(image_path)
    
    cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return vehicle_image 

# for filename in os.listdir('/content/drive/MyDrive/8th/EXECUTION/CARs'):  
for filename in os.listdir(r'F:\Study\Final_Year_Project\backend\CARs'):
  print(filename)  
  plt.figure(figsize=(8,8))
  plt.axis(False)
  plt.imshow(draw_box(r'F:\Study\Final_Year_Project\backend\CARs\{}'.format(filename),CORD[counter]))
  plt.savefig(r'F:\Study\Final_Year_Project\backend\images\{}'.format(s[counter]))
  counter +=1

time.sleep(2)
# my all files starts with "frame" and ends with ".jpg"
fileList1 = glob.glob(r'F:\Study\Final_Year_Project\backend\CARs\*.png')
print("Number of files: ", len(fileList1))

for filePath in fileList1:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)
fileList1 = glob.glob(r'F:\Study\Final_Year_Project\backend\License Plates\*.png')
print("Number of files: ", len(fileList1))

for filePath in fileList1:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)
fileList2 = glob.glob(r'F:\Study\Final_Year_Project\backend\License Plates\ROI\*.png')
print("Number of files: ", len(fileList2))

for filePath in fileList2:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

fileList3 = glob.glob(r'F:\Study\Final_Year_Project\backend\uploads\Img.jpg')

for filePath in fileList3:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)