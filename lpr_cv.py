import cv2
from skimage import filters
import matplotlib.pyplot as plt
image = cv2.imread('image_0005.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobeleddd=filters.sobel(gray_image)
# cv2.imwrite('gray_image.png',gray_image)
blur = cv2.GaussianBlur(gray_image,(5,5),10)
cv2.imshow('blur',blur)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()

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