import numpy as np
import cv2
from pylab import *
import copy
import signal as sig

img1 = cv2.imread("Resources/im_left.png")
img2 = cv2.imread("Resources/im_right.png")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
groundTruthIm = cv2.imread("Resources/disp_left.png")
max_truth_Disp = int(np.max(groundTruthIm) / 3)
Im_Name = "MOEBIUS "
# im = np.vstack((img1, img2))
DisparityIm = np.zeros(np.shape(img1))
DisparityIm = np.float64(DisparityIm)
k = 3
radius = int(np.floor(k / 2))
rows = img1.shape[0]
cols = img1.shape[1]
for i in range(rows):
    for j in np.arange(cols - 1, 0, -1):
        left = j - radius
        right = j + radius +1
        top = i - radius
        bottom = i + radius +1
        if left < 0 or right > cols or top < 0 or bottom > rows:
            continue

        template = copy.deepcopy(img1[top:bottom, left:right])
        template = np.float64(template)

        min_val = np.inf
        min_loc = 0
        for m in np.arange(j - 1, max((j-max_truth_Disp), 0), -1):
            leftTemp = m - radius
            rightTemp = m + radius + 1
            if leftTemp < 0 or rightTemp > cols:
                continue
            else:
                scanLineTemplate = np.float32(copy.deepcopy(img2[top:bottom, leftTemp:rightTemp]))
                SSD = np.sum(np.subtract(scanLineTemplate, template) * np.subtract(scanLineTemplate, template))
                if SSD < min_val:
                    min_val = SSD
                    min_loc = m
        DisparityIm[i][j] = np.abs(j - min_loc)
        # print("Disparity in [I][J] is:" + str(DisparityIm[i][j]) + " i= " + str(i) + " j = " + str(j))


cv2.imwrite(Im_Name + "- SSD Disparity Image - Window Size -  " +str(k) +".jpg", DisparityIm)
cv2.imwrite(Im_Name + "- SSD Disparity Image - Window Size -  " +str(k) +" - better contrast.jpg", DisparityIm * 3)

groundTruthIm = cv2.imread("Resources/disp_left.png")
groundTruthIm = cv2.cvtColor(groundTruthIm, cv2.COLOR_BGR2GRAY)
groundTruthIm = np.float64(groundTruthIm) / 3
groundTruthIm = groundTruthIm[radius:(rows-radius), radius:(cols-radius)]

'''
1 - mean of absolute differences
2 - median
3 - Bad 0.5
4 - Bad 4
'''


temp = DisparityIm[radius:(rows-radius), radius:(cols-radius)]
r, c = np.shape(temp)
print("Window Size: " + str(k))

res = np.mean(np.absolute(temp - groundTruthIm))
print("the mean of absolute differences is: " + str(res))

res = np.median(np.absolute(temp - groundTruthIm))
print("the median of absolute differences is: " + str(res))


res = np.absolute(temp - groundTruthIm)
count = 0
for i in range(r):
    for j in range(c):
        if res[i][j] > 0.5:
            count = count + 1
res = (count / np.size(groundTruthIm)) * 100
print("the bad0.5 is: " + str(res))


res = np.absolute(temp - groundTruthIm)
count = 0
for i in range(r):
    for j in range(c):
        if res[i][j] > 4:
            count = count + 1
res = (count / np.size(groundTruthIm)) * 100
print("the bad4 is: " + str(res))
print("******************************************************")
print(" ")




DisparityIm = np.zeros(np.shape(img1))
DisparityIm = np.float64(DisparityIm)
k = 9
radius = int(np.floor(k / 2))
rows = img1.shape[0]
cols = img1.shape[1]
for i in range(rows):
    for j in np.arange(cols - 1, 0, -1):
        left = j - radius
        right = j + radius +1
        top = i - radius
        bottom = i + radius +1
        if left < 0 or right > cols or top < 0 or bottom > rows:
            continue

        template = copy.deepcopy(img1[top:bottom, left:right])
        template = np.float64(template)

        min_val = np.inf
        min_loc = 0
        for m in np.arange(j - 1, max((j-max_truth_Disp), 0), -1):
            leftTemp = m - radius
            rightTemp = m + radius + 1
            if leftTemp < 0 or rightTemp > cols:
                continue
            else:
                scanLineTemplate = np.float32(copy.deepcopy(img2[top:bottom, leftTemp:rightTemp]))
                SSD = np.sum(np.subtract(scanLineTemplate, template) * np.subtract(scanLineTemplate, template))
                if SSD < min_val:
                    min_val = SSD
                    min_loc = m
        DisparityIm[i][j] = np.absolute(j - min_loc)
        # print("Disparity in [I][J] is:" + str(DisparityIm[i][j]) + " i= " + str(i) + " j = " + str(j))
DisparityImTemp = np.copy(DisparityIm)

cv2.imwrite(Im_Name + "- SSD Disparity Image - Window Size -  " +str(k) +".jpg", DisparityIm)
cv2.imwrite(Im_Name + "- SSD Disparity Image - Window Size -  " +str(k) +" - better contrast.jpg", DisparityIm * 3)

groundTruthIm = cv2.imread("Resources/disp_left.png")
groundTruthIm = cv2.cvtColor(groundTruthIm, cv2.COLOR_BGR2GRAY)
groundTruthIm = np.float64(groundTruthIm) / 3
groundTruthIm = groundTruthIm[radius:(rows-radius), radius:(cols-radius)]

'''
1 - mean of absolute differences
2 - median
3 - Bad 0.5
4 - Bad 4
'''


temp = DisparityIm[radius:(rows-radius), radius:(cols-radius)]
r, c = np.shape(temp)
print("Window Size: " + str(k))

res = np.mean(np.absolute(temp - groundTruthIm))
print("the mean of absolute differences is: " + str(res))

res = np.median(np.absolute(temp - groundTruthIm))
print("the median of absolute differences is: " + str(res))


res = np.absolute(temp - groundTruthIm)
count = 0
for i in range(r):
    for j in range(c):
        if res[i][j] > 0.5:
            count = count + 1
res = (count / np.size(groundTruthIm)) * 100
print("the bad0.5 is: " + str(res))


res = np.absolute(temp - groundTruthIm)
count = 0
for i in range(r):
    for j in range(c):
        if res[i][j] > 4:
            count = count + 1
res = (count / np.size(groundTruthIm)) * 100
print("the bad4 is: " + str(res))
print("******************************************************")




DisparityIm = np.zeros(np.shape(img1))
DisparityIm = np.float64(DisparityIm)
k = 15
radius = int(np.floor(k / 2))
rows = img1.shape[0]
cols = img1.shape[1]
for i in range(rows):
    for j in np.arange(cols - 1, 0, -1):
        left = j - radius
        right = j + radius +1
        top = i - radius
        bottom = i + radius +1
        if left < 0 or right > cols or top < 0 or bottom > rows:
            continue

        template = copy.deepcopy(img1[top:bottom, left:right])
        template = np.float64(template)

        min_val = np.inf
        min_loc = 0
        for m in np.arange(j - 1, max((j-max_truth_Disp), 0), -1):
            leftTemp = m - radius
            rightTemp = m + radius + 1
            if leftTemp < 0 or rightTemp > cols:
                continue
            else:
                scanLineTemplate = np.float32(copy.deepcopy(img2[top:bottom, leftTemp:rightTemp]))
                SSD = np.sum(np.subtract(scanLineTemplate, template) * np.subtract(scanLineTemplate, template))
                if SSD < min_val:
                    min_val = SSD
                    min_loc = m
        DisparityIm[i][j] = np.absolute(j - min_loc)
        # print("Disparity in [I][J] is:" + str(DisparityIm[i][j]) + " i= " + str(i) + " j = " + str(j))
DisparityImTemp = np.copy(DisparityIm)

cv2.imwrite(Im_Name + "- SSD Disparity Image - Window Size -  " +str(k) +".jpg", DisparityIm)
cv2.imwrite(Im_Name + "- SSD Disparity Image - Window Size -  " +str(k) +" - better contrast.jpg", DisparityIm * 3)
groundTruthIm = cv2.imread("Resources/disp_left.png")
groundTruthIm = cv2.cvtColor(groundTruthIm, cv2.COLOR_BGR2GRAY)
groundTruthIm = np.float64(groundTruthIm) / 3
groundTruthIm = groundTruthIm[radius:(rows-radius), radius:(cols-radius)]

'''
1 - mean of absolute differences
2 - median
3 - Bad 0.5
4 - Bad 4
'''


temp = DisparityIm[radius:(rows-radius), radius:(cols-radius)]
r, c = np.shape(temp)
print("Window Size: " + str(k))

res = np.mean(np.absolute(temp - groundTruthIm))
print("the mean of absolute differences is: " + str(res))

res = np.median(np.absolute(temp - groundTruthIm))
print("the median of absolute differences is: " + str(res))


res = np.absolute(temp - groundTruthIm)
count = 0
for i in range(r):
    for j in range(c):
        if res[i][j] > 0.5:
            count = count + 1
res = (count / np.size(groundTruthIm)) * 100
print("the bad0.5 is: " + str(res))


res = np.absolute(temp - groundTruthIm)
count = 0
for i in range(r):
    for j in range(c):
        if res[i][j] > 4:
            count = count + 1
res = (count / np.size(groundTruthIm)) * 100
print("the bad4 is: " + str(res))
print("******************************************************")
print(" ")
