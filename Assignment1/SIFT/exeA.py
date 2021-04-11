import cv2
import numpy as np

img1 = cv2.imread("Resources/UoH.jpg")

sift = cv2.SIFT_create()
kp1, descriptors = sift.detectAndCompute(img1, None)

count = 1
for kp in kp1:
    if count % 5 == 0:
        count += 1
        pass
    else:
        angle = kp.angle
        size = kp.size
        x_cord = round(kp.pt[0])
        y_cord = round(kp.pt[1])
        x1 = round(kp.pt[0] + (kp.size * np.cos(kp.angle)))
        y1 = round(kp.pt[1] + kp.size * np.sin(kp.angle))
        cv2.arrowedLine(img1,(x_cord,y_cord),(x1,y1),(255,0,0), 1)
        count += 1


img3 = cv2.drawKeypoints(img1,kp1, None, (0,0,255))
cv2.imshow("Key Points as simon wants", img1)
cv2.imshow("Key Points that simon doesn't approve", img3)
cv2.waitKey(0)
