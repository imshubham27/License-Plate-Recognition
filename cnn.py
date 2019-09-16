from PIL import Image
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model

#FOR CAR DETECTION
classifier = Sequential()

#Convolution
classifier.add(Conv2D(32, (5, 5), input_shape = (128, 128, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
#adam is stochastic gradient optimizer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen= ImageDataGenerator(rescale = 1./255)

training_set =train_datagen.flow_from_directory('training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 2045,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 593)
#SAVING THE MODEL
classifier.save('Car_Recognition.h5')
#from keras.preprocessing import image
#test_image = image.load_img('image_0481.jpg', target_size = (128, 128))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict_proba(test_image)
#training_set.class_indices
#print(result)
#if result[0][0] == 1:
#    prediction = 'non-car'
#else:
#    prediction = 'car'
      
#FOR SLIDING WINDOW & CLASSIFYING
# import the necessary packages
import imutils
import cv2
import time
from keras.preprocessing import image

image = cv2.imread('i (4).jpg')
height = image.shape[0]
width = image.shape[1]

def pyramid(image, scale=3, minSize=(32, 32)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

c=0
def model(win_image):
        global c
        print("IN WINDOW")
        test_img = cv2.resize(win_image,(128,128))
        test_img = np.expand_dims(test_img, axis = 0)
        result = classifier.predict(test_img)
        training_set.class_indices
        if result[0][0] ==0:
            c=c+1
            prediction = 'CAR'
            cv2.imwrite('image({}).jpg'.format(c),win_image)
        else:
            prediction = 'Non-Car'
        print(prediction)
        print(result)
        
# loop over the image pyramid
for (i, resized) in enumerate(pyramid(image, scale=3)):
	# show the resized image
	cv2.imshow('image',resized)
	cv2.waitKey(0)
# close all windows
cv2.destroyAllWindows()

(winW, winH) = (int(width/4),int(height/4))

# loop over the image pyramid
for resized in pyramid(image, scale=3):
	# loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue                 
		# since we do not have a classifier, we'll just draw the window
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        model(window)
        cv2.imshow("SLIDING WINDOW",clone)
        cv2.waitKey(1)
        time.sleep(0.025)
        
cv2.destroyAllWindows()

#FOR PLATE DETECTION
classifier1 = Sequential()

#Convolution
classifier1.add(Conv2D(32, (5, 5), input_shape = (128, 128, 3), activation = 'relu'))

#Pooling
classifier1.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier1.add(Conv2D(64, (5, 5), activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
classifier1.add(Flatten())

#Full connection
classifier1.add(Dense(units = 128, activation = 'relu'))
classifier1.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
#adam is stochastic gradient optimizer
classifier1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen1= ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen1= ImageDataGenerator(rescale = 1./255)

training_set1 =train_datagen1.flow_from_directory('training1_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set1 = test_datagen1.flow_from_directory('test1_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier1.fit_generator(training_set1,
                         steps_per_epoch = 1714,
                         epochs = 3,
                         validation_data = test_set1,
                         validation_steps = 328)

#SAVING THE MODEL
classifier1.save('License_Plate_Recognition.h5')
#from keras.preprocessing import image
#test_image = image.load_img('image_0481.jpg', target_size = (128, 128))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict_proba(test_image)
#training_set.class_indices
#print(result)
#if result[0][0] == 1:
#    prediction = 'non-car'
#else:
#    prediction = 'car'


#FOR SLIDING WINDOW & CLASSIFYING
# import the necessary packages
import cv2
import time
from keras.preprocessing import image

image1 = cv2.imread('image(133).jpg')
height1 = image1.shape[0]
width1 = image1.shape[1]

def pyramid1(image1, scale=3, minSize=(16, 16)):
	# yield the original image
	yield image1

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image1.shape[1] / scale)
		image1 = imutils.resize(image1, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image1.shape[0] < minSize[1] or image1.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image1

def sliding_window1(image1, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image1.shape[0], stepSize):
		for x in range(0, image1.shape[1], stepSize):
			# yield the current window
			yield (x, y, image1[y:y + windowSize[1], x:x + windowSize[0]])

e=0
def model1(win_image1):
        global e
        print("IN WINDOW")
        test_img1 = cv2.resize(win_image1,(128,128))
        test_img1 = np.expand_dims(test_img1, axis = 0)
        result1 = classifier1.predict(test_img1)
        training_set1.class_indices
        if result1[0][0] ==0:
            e=e+1
            prediction1 = 'License Plate Region'
            cv2.imwrite('image({}).jpg'.format(e),win_image1)
        else:
            prediction1 = 'Non-License Plate Region'
        print(prediction1)
        print(result1)
        
# loop over the image pyramid
for (i, resized1) in enumerate(pyramid1(image1, scale=3)):
	# show the resized image
	cv2.imshow('image',resized1)
	cv2.waitKey(0)
# close all windows
cv2.destroyAllWindows()

(winW1, winH1) = (int(width1/4), (int(height1/4)))

# loop over the image pyramid
for resized1 in pyramid1(image1, scale=3):
	# loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized1, stepSize=16, windowSize=(winW1, winH1)):
		# if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH1 or window.shape[1] != winW1:
            continue                 
		# since we do not have a classifier, we'll just draw the window
        clone1 = resized1.copy()
        cv2.rectangle(clone1, (x, y), (x + winW1, y + winH1), (0, 255, 0), 2)
        model1(window)
        cv2.imshow("SLIDING WINDOW",clone1)
        cv2.waitKey(1)
        time.sleep(0.025)
        
cv2.destroyAllWindows()

#BUILDING DIGIT RECOGNIZER

classifier2 = Sequential()

#Convolution
classifier2.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
classifier2.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier2.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier2.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
classifier2.add(Flatten())

#Full connection
classifier2.add(Dense(units = 128, activation = 'relu'))
classifier2.add(Dense(units = 37, activation = 'softmax'))

# Compiling the CNN
#adam is stochastic gradient optimizer
classifier2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen2= ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen2= ImageDataGenerator(rescale = 1./255)

training_set2 =train_datagen2.flow_from_directory('training2_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set2 = test_datagen2.flow_from_directory('test2_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier2.fit_generator(training_set2,
                         steps_per_epoch = 7865,
                         epochs = 3,
                         validation_data = test_set2,
                         validation_steps = 2082)

test_steps_per_epoch2 = np.math.ceil(test_set2.samples / test_set2.batch_size)

predictions2 = classifier2.predict_generator(test_set2, steps=test_steps_per_epoch2)
true_classes2 = test_set2.classes
class_labels2 = list(test_set2.class_indices.keys())   
# Get most likely class
predicted_classes2 = np.argmax(predictions2, axis=1)
report = classification_report(true_classes2, predicted_classes2, target_names=class_labels2)
print(report)    

#SAVING THE MODEL
classifier2.save('Character_Recognition.h5')

from keras.preprocessing import image
test_image2 = image.load_img('image(20).jpg', target_size = (64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)
result2 = classifier2.predict_proba(test_image2)
training_set2.class_indices
if result2[0][0] == 1:
    prediction = 'NCD'
if result2[0][1] == 1:
    prediction = 0
if result2[0][2] == 1:
    prediction = 1
if result2[0][3] == 1:
    prediction = 2
if result2[0][4] == 1:
    prediction = 3
if result2[0][5] == 1:
    prediction = 4
if result2[0][6] == 1:
    prediction = 5
if result2[0][7] == 1:
    prediction = 6
if result2[0][8] == 1:
    prediction = 7
if result2[0][9] == 1:
    prediction = 8
if result2[0][10] == 1:
    prediction = 9
if result2[0][11] == 1:
    prediction = 'A'
if result2[0][12] == 1:
    prediction = 'B'
if result2[0][13] == 1:
    prediction = 'C'
if result2[0][14] == 1:
    prediction = 'D'
if result2[0][15] == 1:
    prediction = "E"
if result2[0][16] == 1:
    prediction = 'F'
if result2[0][17] == 1:
    prediction = 'G'
if result2[0][18] == 1:
    prediction = 'H'
if result2[0][19] == 1:
    prediction = 'I'
if result2[0][20] == 1:
    prediction = 'J'
if result2[0][21] == 1:
    prediction = 'K'
if result2[0][22] == 1:
    prediction = 'L'
if result2[0][23] == 1:
    prediction = 'M'
if result2[0][24] == 1:
    prediction = 'N'
if result2[0][25] == 1:
    prediction = 'O'
if result2[0][26] == 1:
    prediction = 'P'
if result2[0][27] == 1:
    prediction = 'Q'
if result2[0][28] == 1:
    prediction = 'R'
if result2[0][29] == 1:
    prediction = 'S'
if result2[0][30] == 1:
    prediction = 'T'
if result2[0][31] == 1:
    prediction = 'U'
if result2[0][32] == 1:
    prediction = 'V'
if result2[0][33] == 1:
    prediction = 'W'
if result2[0][34] == 1:
    prediction = 'X'
if result2[0][35] == 1:
    prediction = 'Y'
if result2[0][36] == 1:
    prediction = 'Z'
print(result2)

#PREPROCESSING AND FINDING THE CHARACTER
import cv2 
import numpy as np 
# Let's load a simple image with 3 black squares 
image3 = cv2.imread('I00000.jpg')

# Grayscale 
gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY) 
blur = cv2.GaussianBlur(gray, (11, 11), 0)
#cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
#img = cv2.bitwise_not(img)
cv2.imshow("BLURRED",image3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find Canny edges 
edged = cv2.Canny(blur, 30, 200) 
cv2.waitKey(0) 
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged, 
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
n=len(contours)
area=[]

for i in range(n):
    area.append(cv2.contourArea(contours[i]))
    
area=np.asarray(area)
print(area)
maxpos =np.where(area == np.amax(area))
k=np.asscalar(maxpos[0])

cv2.imshow('Canny Edges After Contouring', edged) 
cv2.waitKey(0) 

# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(image3, contours, -1, (0, 255, 0), 3) 

cv2.imshow('Contours', image3) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
#for ctr in contours:
    # Get bounding box
x, y, w, h = cv2.boundingRect(contours[k])
    # Getting ROI
roi = image3[y:y+h, x:x+w]

cv2.imshow('max',roi)
cv2.imwrite('max_contour.jpg', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

#FOR DESKEWING THE IMAGE
#coords = np.column_stack(np.where(img > 0))
#angle = cv2.minAreaRect(coords)[-1]
 
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
#if angle < -45:
#	angle = -(90 + angle)
 
# otherwise, just take the inverse of the angle to make
# it positive
#else:
#	angle = -angle
# rotate the image to deskew it
#(h, w) = img.shape[:2]
#center = (w // 2, h // 2)
#M = cv2.getRotationMatrix2D(center, angle, 1.0)
#rotated = cv2.warpAffine(img, M, (w, h),
#	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# draw the correction angle on the image so we can validate it
#cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
#	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
# show the output image
#print("[INFO] angle: {:.3f}".format(angle))
#cv2.imshow("Input", img)
#cv2.waitKey(0)

#contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
#cv2.imshow("contours", img)
#cv2.waitKey(0)
#d=0
#for ctr in contours:
#    # Get bounding box
#    x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
#    roi = img[y:y+h, x:x+w]

#    cv2.imshow('character: %d'%d,roi)
#    cv2.imwrite('character_%d.png'%d, roi)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#   d+=1
    
    
    

import cv2
img = cv2.imread('max_contour.jpg', 0)
#img = cv2.GaussianBlur(img, (11, 11), 0)
cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
img = cv2.bitwise_not(img)
contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
cv2.imshow("contours", img)
cv2.waitKey(0)
d=0
for ctr in contours:
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
    roi = img[y:y+h, x:x+w]

    cv2.imshow('character: %d'%d,roi)
    cv2.imwrite('character_%d.png'%d, roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    d+=1
