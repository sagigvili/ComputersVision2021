import numpy as np
import cv2

im_array = []

for i in range(0, 420):
    im_array.append(cv2.imread("alley_2/im "+str(i)+".jpg"))

height, width, e = np.shape(im_array[0])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.avi', fourcc, 10, (width, height))

for j in range(0, 420):
    video.write(im_array[j])
video.release()