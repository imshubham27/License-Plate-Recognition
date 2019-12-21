# -*- coding: utf-8 -*-
"""segmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T4dikXYED4iBYX90WQJKUZgqHmk-GCrq
"""

from google.colab import drive
drive.mount('/content/drive')

#pip install scikit-image

import cv2
import numpy as np
import math
from skimage import img_as_ubyte
from skimage import io
from skimage.color import rgb2gray
from google.colab.patches import cv2_imshow

img = io.imread('/content/drive/My Drive/car15.png')
print(img.shape)
#img=cv2.resize
#cv2_imshow(img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)
cv2_imshow(value)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)

cv2_imshow(topHat)
cv2_imshow(blackHat)

add = cv2.add(value, topHat)
subtract = cv2.subtract(add, blackHat)

cv2_imshow(subtract)
cv2_imshow(add)

blur = cv2.GaussianBlur(subtract, (3,3), 0)

cv2_imshow(blur)

thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
cv2_imshow(thresh)

import math
import cv2


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

        self.centerX = (self.boundingRectX + self.boundingRectX + self.boundingRectWidth) / 2
        self.centerY = (self.boundingRectY + self.boundingRectY + self.boundingRectHeight) / 2

        self.diagonalSize = math.sqrt((self.boundingRectWidth ** 2) + (self.boundingRectHeight ** 2))

        self.aspectRatio = float(self.boundingRectWidth) / float(self.boundingRectHeight)


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

cv2MajorVersion = cv2.__version__.split(".")[0]

if int(cv2MajorVersion) >= 4:
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
else:
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

height, width = thresh.shape

imageContours = np.zeros((height, width, 3), dtype=np.uint8)

possibleChars = []
countOfPossibleChars = 0

for i in range(0, len(contours)):

    # draw contours based on actual found contours of thresh image
    cv2.drawContours(imageContours, contours, i, (255, 255, 255))

    # retrieve a possible char by the result ifChar class give us
    possibleChar = ifChar(contours[i])

    # by computing some values (area, width, height, aspect ratio) possibleChars list is being populated
    if checkIfChar(possibleChar) is True:
        countOfPossibleChars = countOfPossibleChars + 1
        possibleChars.append(possibleChar)

cv2_imshow( imageContours)

imageContours = np.zeros((height, width, 3), np.uint8)

ctrs = []

for char in possibleChars:
    ctrs.append(char.contour)

cv2.drawContours(imageContours, ctrs, -1, (255, 255, 255))

cv2_imshow(imageContours)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
dilated = cv2.dilate(imageContours, kernel, iterations=4)
cv2_imshow(dilated)

from skimage.color import rgb2gray
grayscale = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
cv2_imshow(grayscale)
thresh1 = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
cnts,_ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if len(cnts) == 2 else cnts[1]

ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if (area > 1000) and (area < 100000):
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 3)
        # ROI = image[y:y+h, x:x+w]
        # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        # ROI_number += 1

#cv2.imshow('thresh', thresh)
cv2_imshow(dilated)
cv2_imshow(img)

plates_list = []
listOfListsOfMatchingChars = []

for possibleC in possibleChars:

    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    def matchingChars(possibleC, possibleChars):
        listOfMatchingChars = []

        # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
        # then we should not include it in the list of matches b/c that would end up double including the current char
        # so do not add to list of matches and jump back to top of for loop
        for possibleMatchingChar in possibleChars:
            if possibleMatchingChar == possibleC:
                continue

            # compute stuff to see if chars are a match
            dBetweenChars = distanceBetweenChars(possibleC, possibleMatchingChar)

            aBetweenChars = angleBetweenChars(possibleC, possibleMatchingChar)

            changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(
                possibleC.boundingRectArea)

            changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(
                possibleC.boundingRectWidth)

            changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(
                possibleC.boundingRectHeight)

            # check if chars match
            if dBetweenChars < (possibleC.diagonalSize * 2.5) and \
                    aBetweenChars < 12.0 and \
                    changeInArea < 0.5 and \
                    changeInWidth < 0.8 and \
                    changeInHeight < 0.2:
                listOfMatchingChars.append(possibleMatchingChar)

        return listOfMatchingChars


    # here we are re-arranging the one big list of chars into a list of lists of matching chars
    # the chars that are not found to be in a group of matches do not need to be considered further
    listOfMatchingChars = matchingChars(possibleC, possibleChars)

    listOfMatchingChars.append(possibleC)

    # if current possible list of matching chars is not long enough to constitute a possible plate
    # jump back to the top of the for loop and try again with next char
    if len(listOfMatchingChars) < 3:
        continue

    # here the current list passed test as a "group" or "cluster" of matching chars
    listOfListsOfMatchingChars.append(listOfMatchingChars)

    # remove the current list of matching chars from the big list so we don't use those same chars twice,
    # make sure to make a new big list for this since we don't want to change the original big list
    listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

    recursiveListOfListsOfMatchingChars = []

    for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
        listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

    break

imageContours = np.zeros((height, width, 3), np.uint8)

for listOfMatchingChars in listOfListsOfMatchingChars:
    contoursColor = (255, 0, 255)

    contours = []

    for matchingChar in listOfMatchingChars:
        contours.append(matchingChar.contour)

    cv2.drawContours(imageContours, contours, -1, contoursColor)

cv2_imshow(imageContours)

for listOfMatchingChars in listOfListsOfMatchingChars:
    possiblePlate = PossiblePlate()

    # sort chars from left to right based on x position
    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)

    # calculate the center point of the plate
    plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX) / 2.0
    plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY) / 2.0

    plateCenter = plateCenterX, plateCenterY

    # calculate plate width and height
    plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)

    totalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight

    averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

    plateHeight = int(averageCharHeight * 1.5)

    # calculate correction angle of plate region
    opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY

    hypotenuse = distanceBetweenChars(listOfMatchingChars[0],
                                                listOfMatchingChars[len(listOfMatchingChars) - 1])
    correctionAngleInRad = math.asin(opposite / hypotenuse)
    correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

    height, width, numChannels = img.shape

    # rotate the entire image
    imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

    # crop the image/plate detected
    imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))

    # copy the cropped plate image into the applicable member variable of the possible plate
    possiblePlate.Plate = imgCropped

    # populate plates_list with the detected plate
    if possiblePlate.Plate is not None:
        plates_list.append(possiblePlate)

    # draw a ROI on the original image
    for i in range(0, len(plates_list)):
        # finds the four vertices of a rotated rect - it is useful to draw the rectangle.
        p2fRectPoints = cv2.boxPoints(plates_list[i].rrLocationOfPlateInScene)
        print(p2fRectPoints)

        # roi rectangle colour
        rectColour = (0, 255, 0)

        cv2.line(imageContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
        cv2.line(imageContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
        cv2.line(imageContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
        cv2.line(imageContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

        cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
        cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
        cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
        cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)
        #cv2_imshow(imageContours)
        # cv2.imwrite(temp_folder + '11 - detected.png', imageContours)

        #cv2_imshow(img)
        # cv2.imwrite(temp_folder + '12 - detectedOriginal.png', img)
        LP=plates_list[i].Plate
        cv2_imshow(plates_list[i].Plate)
        cv2.imwrite('/content/drive/My Drive/plate10/a1.png', plates_list[i].Plate)

