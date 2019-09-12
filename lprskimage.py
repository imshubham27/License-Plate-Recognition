import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage import io
from skimage.color import rgb2gray
from skimage import filters

#original =io.imread_collection(r'C:\Users\Shubham\Desktop\Cars\*.jpg')
original =io.imread('image_0005.jpg')
grayscale = rgb2gray(original)
io.imshow(grayscale)
gau_img =filters.gaussian(grayscale,sigma=1)
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
plt.show()