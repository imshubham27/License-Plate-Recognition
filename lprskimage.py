"""import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage import io
from skimage.color import rgb2gray
from skimage import filters

#original =io.imread_collection(r'C:\Users\Shubham\Desktop\Cars\*.jpg')
original =io.imread('image_0005.jpg')
grayscale = rgb2gray(original)
io.imshow(grayscale)
gau_img =filters.gaussian(grayscale,sigma=10, truncate=1/5)
io.imshow(gau_img)
edges=filters.sobel(gau_img)
io.imshow(edges)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].io.imshow_collection(original)
ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(gau_img, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

fig.tight_layout()
plt.show()"""

import cv2
from skimage import filters
import matplotlib.pyplot as plt
image = cv2.imread('image_0005.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobeleddd=filters.sobel(gray_image)
# cv2.imwrite('gray_image.png',gray_image)
blur = cv2.GaussianBlur(gray_image,(5,5),0)

sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)
cv2.imshow('sobelx',sobeleddd)
sobeled=cv2.add(sobelx,sobely)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows() 
#cv2.imshow('color_image',image)
plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobeled,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([]) 
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()  