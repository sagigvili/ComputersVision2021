import cv2
import numpy as np
from numpy import random


def keyoints_matching_ratio(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    newIm = np.hstack((img1, img2))

    ''' finding key points and their descriptors '''
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)


    ''' finding the distance MXN matrix between each descriptors pair'''
    rows = len(desc1)
    cols = len(desc2)
    distanceOfDescriptors = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            distanceOfDescriptors[i][j] = cv2.norm(desc1[i] - desc2[j], cv2.NORM_L2)


    ''' Matching the key points  based over the distances matrix '''
    bestMatch = np.inf
    secondBestMatch = np.inf
    matches = []
    tempIndex = None
    indexes = []
    count = 0
    for i in range(rows):
        bestMatch = np.inf
        secondBestMatch = np.inf
        for j in range(cols):
            if distanceOfDescriptors[i][j] < bestMatch:
                bestMatch = distanceOfDescriptors[i][j]
                tempIndex = j

        for k in range(cols):
            if k == tempIndex:
                continue
            elif distanceOfDescriptors[i][k] < secondBestMatch:
                secondBestMatch = distanceOfDescriptors[i][k]

        if bestMatch < 0.8 * secondBestMatch:
            temp2 = (kp1[i], kp2[tempIndex])
            matches.append(temp2)

        else:
            continue

    shapeImg1 = np.shape(img1)
    newRowOffset = shapeImg1[0]
    newColsOffset = shapeImg1[1]
    c = 0
    for match in matches:
        c += 1
        if (c % 10) == 0:
            x1 = round(match[0].pt[0])
            y1 = round(match[0].pt[1])
            x2 = round(match[1].pt[0])
            y2 = round(match[1].pt[1])
            x2 += newColsOffset
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            count += 1
            cv2.line(newIm, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(newIm, (x1, y1), 4, (255, 0, 0), 1)
            cv2.circle(newIm, (x2, y2), 4, (255, 0, 0), 1)
    print("number of matches: " + str(c))
    return newIm
