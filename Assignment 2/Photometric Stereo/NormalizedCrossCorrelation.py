import numpy as np
import cv2
from pylab import *

img1 = cv2.imread("Resources/im_left.png")
img2 = cv2.imread("Resources/im_right.png")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# im = np.vstack((img1, img2))
DisparityIm = np.zeros(np.shape(img1))
k = 15
radius = int(np.floor(k / 2))
rows = img1.shape[0]
cols = img1.shape[1]
for i in range(rows):
    for j in range(cols):
        left = j - radius
        right = j + radius + 1
        top = i - radius
        bottom = i + radius + 1
        # if left < 0 or right > cols or top < 0 or bottom > rows:
        #     continue
        if left < 0:
            left = 0
        if right > cols:
            right = cols
        if top < 0:
            top = 0
        if bottom > rows:
            bottom = rows
        # else:
        template = img1[top:bottom, left:right]
        scanLineTemplate = img2[top:bottom, 0:cols]
        # maxCorr = 0
        # maxLocation = None
        NCC = cv2.matchTemplate(template, scanLineTemplate, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(NCC)
        temp = max_loc[0]
        if temp == j:
            temp = temp + 0.00001
        DisparityIm[i][j] = 20 / (cv2.norm((j - temp), cv2.NORM_L2))
        # for m in range(cols):
        #     leftTemp = m - radius
        #     rightTemp = m + radius + 1
        #     if leftTemp < 0 or rightTemp > cols:
        #         continue
        #     else:
        #         scanLineTemplate = img2[top:bottom, leftTemp:rightTemp]
        #         NCC = cv2.matchTemplate(template, scanLineTemplate, cv2.TM_CCORR_NORMED)
        #         # a = np.matrix.flatten(template)
        #         # b = np.matrix.flatten(scanLineTemplate)
        #         # normA = a / np.linalg.norm(a)
        #         # normB = b / np.linalg.norm(b)
        #         # NCC = np.dot(a, b) / np.dot(normA, normB)
        #         if NCC[0][0] < minDist:
        #             minDist = NCC[0][0]
        #             minLocation = (i, k)

        # minDist = np.inf
        # temp = minLocation[1]
        # if j - temp == 0:
        #     temp = temp + 0.00001
        # DisparityIm[i][j] = np.round(255 / (np.sqrt((j - temp)**2)))
        print("Disparity in [I][J] is:" + str(DisparityIm[i][j]) + " i= " + str(i) + " j = " + str(j))
        # minLocation = None

# stereo = cv2.StereoBM_create(numDisparities=64, blockSize=5)
# disparity = stereo.compute(img1,img2)
# plt.imshow(disparity, 'gray')
# plt.show()

# cv2.imshow("sasi altogether", im)
# cv2.imshow("sasi 1", img1)
# newIm = np.zeros(np.shape(DisparityIm))
# cv2.normalize(DisparityIm, newIm, 0, 1, cv2.NORM_MINMAX)
DisparityImTemp = 1 - DisparityIm
cv2.imshow("sasi 2", DisparityImTemp)
cv2.waitKey(0)

groundTruthIm = cv2.imread("Resources/disp_left.png")
groundTruthIm = cv2.cvtColor(groundTruthIm, cv2.COLOR_BGR2GRAY)

'''
1 - mean of absolute differences
2 - median
3 - Bad 0.5
4 - Bad 4
'''
measureType = 4

if measureType == 1:
    temp = DisparityIm / 20
    temp = np.round(((temp + 1) / 2 ) * 255)
    res = np.mean(np.absolute(temp - groundTruthIm))
    print("the mean of absolute differences is: " + str(res))

elif measureType == 2:
    temp = DisparityIm / 20
    temp = np.round(((temp + 1) / 2 ) * 255)
    res = np.median(np.absolute(temp - groundTruthIm))
    print("the median of absolute differences is: " + str(res))

elif measureType == 3:
    temp = DisparityIm / 20
    temp = ((temp + 1) / 2) * 255
    res = np.absolute(temp - groundTruthIm)
    count = 0
    for i in range(rows):
        for j in range(cols):
            if res[i][j] > 0.5:
                count = count + 1
    res = (count / np.size(groundTruthIm)) * 100
    print("the bad0.5 is: " + str(res))


elif measureType == 4:
    temp = DisparityIm / 20
    temp = ((temp + 1) / 2) * 255
    res = np.absolute(temp - groundTruthIm)
    count = 0
    for i in range(rows):
        for j in range(cols):
            if res[i][j] > 4:
                count = count + 1
    res = (count / np.size(groundTruthIm)) * 100
    print("the bad4 is: " + str(res))
