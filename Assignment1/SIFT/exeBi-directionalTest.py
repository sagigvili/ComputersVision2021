import cv2
import numpy as np
from numpy import random

img1 = cv2.imread("Resources/pair1_imageA.jpg")
img2 = cv2.imread("Resources/pair1_imageB.jpg")

newIm = np.hstack((img1, img2))

''' finding key points and their descriptors '''
sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(desc1,desc2,k=2)
''' finding the distance MXN matrix between each descriptors pair'''
rows = len(desc1)
cols = len(desc2)
distanceOfDescriptors = np.zeros((rows, cols))

# i = 0
# j = 0
# for descriptor1 in desc1:
#     for descriptor2 in desc2:
#         temp = abs(descriptor1 - descriptor2)
#         # temp1 = np.sum(temp)
#         # temp1 = np.sqrt(temp1)
#         # temp1 = cv2.norm((descriptor1, descriptor2), cv2.NORM_L2)
#         distanceOfDescriptors[i][j] = np.sum(temp)
#         j += 1
#     j = 0
#     i += 1


for i in range(rows):
    for j in range(cols):
        temp = (desc1[i] - desc2[j]) * (desc1[i] - desc2[j])
        temp = np.sum(temp)
        distanceOfDescriptors[i][j] = np.sqrt(temp)

''' Matching the key points  based over the distances matrix '''
bestMatch = 10000000000000000000
secondBestMatch = 10000000000000000000
matches = []
tempIndex = None
tempBiDirectionalIndex = None
indexes = []
count = 0
oneDirection = []
oppDirection = []
for i in range(rows):
    # if count > 13:
    #     break
    for j in range(cols):
        if distanceOfDescriptors[i][j] < bestMatch:
            bestMatch = distanceOfDescriptors[i][j]
            tempIndex = j
            print(" temp index is: " + str(tempIndex))
    temp = (i , tempIndex)
    oneDirection.append(temp)
    # for k in range(rows):
    #     if distanceOfDescriptors[k][tempIndex] < secondBestMatch:
    #         secondBestMatch = distanceOfDescriptors[k][tempIndex]
    #         tempBiDirectionalIndex = k
    # if i == k:
    #     indexes.append(tempIndex)
    #     temp2 = (kp1[i], kp2[tempIndex])
    #     matches.append(temp2)
    #     print("a match has been found: i = " + str(i) + " k = " + str(k) + " j = " + str(tempIndex))
bestMatch = 10000000000000000000
for j in range (cols):
    for i in range(rows):
        if distanceOfDescriptors[i][j] < bestMatch:
            bestMatch = distanceOfDescriptors[i][j]
            tempIndex = i
            print(" temp index is: " + str(tempIndex))
    temp = (tempIndex, j)
    oppDirection.append(temp)

for direct1 in oneDirection:
    for oppDirect1 in oppDirection:
        if (direct1[0] == oppDirect1[0]) and (direct1[1] == oppDirect1[1]):
            temp1 = (kp1[direct1[0]], kp2[direct1[1]])
            matches.append(temp1)




shapeImg1 = np.shape(img1)
newRowOffset = shapeImg1[0]
newColsOffset = shapeImg1[1]
for match in matches:
    x1 = round(match[0].pt[0])
    y1 = round(match[0].pt[1])
    x2 = round(match[1].pt[0])
    y2 = round(match[1].pt[1])
    x2 +=  newColsOffset
    # y2 += newColsOffset
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    count += 1
    cv2.line(newIm, (x1, y1), (x2, y2), (0, 0, 255), 1)
    print("x1 = " + str(x1) + " y1 = " + str(y1) + " x2 = " + str(x2) + " y2 = " + str(y2))
print(str(count))
cv2.imshow("SIFT", newIm)
# cv2.imshow("Key Points that simon doesn't approve", img3)
cv2.waitKey(0)
