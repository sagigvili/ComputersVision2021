import cv2
import numpy as np

img = cv2.imread("Resources/image003.jpg")

''' Edge Detection by Canny '''
low_threshold = 350
high_threshold = 550
imgCanny = cv2.Canny(img, low_threshold, high_threshold)

''' Gradient Direction '''
sobelx = cv2.Sobel(imgCanny, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(imgCanny, cv2.CV_64F, 0, 1, ksize=3)
EdgeGradientDirection = sobely
rows, cols = np.shape(sobelx)
for i in range(rows):
    for j in range(cols):
        if sobelx[i][j] == 0:
            sobelx[i][j] = 0.00000000000001
        EdgeGradientDirection[i][j] = np.arctan(sobely[i][j] / sobelx[i][j])




cv2.imshow("check1", sobelx)
cv2.imshow("check2", sobely)
cv2.imshow("check3", EdgeGradientDirection)
cv2.waitKey(0)