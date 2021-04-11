import cv2
import numpy as np
from numpy import random

img1 = cv2.imread("Resources/pair3_imageA.jpg")
img2 = cv2.imread("Resources/pair3_imageB.jpg")

newIm = np.hstack((img1, img2))

''' finding key points and their descriptors '''
sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(desc1,desc2,k=2)
''' finding the distance MXN matrix between each descriptors pair'''
rows = len(kp1)
cols = len(kp2)
distanceOfDescriptors = np.zeros((rows, cols))
i = 0
j = 0
for descriptor1 in desc1:
    for descriptor2 in desc2:
        temp = (descriptor1 - descriptor2) * (descriptor1 - descriptor2)
        temp1 = np.sum(temp)
        temp1 = np.sqrt(temp1)
        # temp1 = cv2.norm((descriptor1, descriptor2), cv2.NORM_L2)
        distanceOfDescriptors[i][j] = temp1
        j += 1
    j = 0
    i += 1

''' Matching the key points  based over the distances matrix '''
bestMatch = 10000000000000000000
secondBestMatch = 10000000000000000000
matches = []
tempIndex = None
indexes = []
count = 0
for i in range(rows):
    # if count > 13:
    #     break
    for j in range(cols):
        if distanceOfDescriptors[i][j] < bestMatch:
            bestMatch = distanceOfDescriptors[i][j]
            tempIndex = j
            print(" temp index is: " + str(tempIndex))
    for k in range(cols):
        if k == tempIndex:
            continue
        elif distanceOfDescriptors[i][j] < secondBestMatch:
            secondBestMatch = distanceOfDescriptors[i][j]

    if bestMatch < 0.8 * secondBestMatch:
        check = 0
        for ind in indexes:
            if (ind == tempIndex):
                check = 1
        if check == 1:
            continue
        else:
            indexes.append(tempIndex)
            temp2 = (kp1[i], kp2[tempIndex])
            matches.append(temp2)

    else:
        continue

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
cv2.imshow("Key Points as simon wants", newIm)
# cv2.imshow("Key Points that simon doesn't approve", img3)
cv2.waitKey(0)
