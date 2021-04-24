import cv2
import numpy as np
import random


def run_ransac(img1, img2):
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


    ############################## Ratio Test #####################################
    bestMatch = np.inf
    secondBestMatch = np.inf
    matches = []
    tempIndex = None
    indexes = []
    count = 0
    locations = []
    for i in range(rows):
        bestMatch = np.inf
        secondBestMatch = np.inf
        for j in range(cols):
            if distanceOfDescriptors[i][j] < bestMatch:
                bestMatch = distanceOfDescriptors[i][j]
                tempIndex = j
                print(" temp index is: " + str(tempIndex))
        for k in range(cols):
            if k == tempIndex:
                continue
            elif distanceOfDescriptors[i][k] < secondBestMatch:
                secondBestMatch = distanceOfDescriptors[i][k]

        if bestMatch < 0.8 * secondBestMatch:
            p1 = kp1[i].pt
            p2 = kp2[tempIndex].pt
            temp2 = (kp1[i], kp2[tempIndex])
            matches.append(temp2)
            locations.append((p1, p2))

        else:
            continue
    ############################## Ratio Test #####################################


    ############################## Drawing Matches #####################################
    # shapeImg1 = np.shape(img1)
    # newRowOffset = shapeImg1[0]
    # newColsOffset = shapeImg1[1]
    # c = 0
    # for match in matches:
    #     c +=1
    #     if (c % 1) == 0:
    #         x1 = round(match[0].pt[0])
    #         y1 = round(match[0].pt[1])
    #         x2 = round(match[1].pt[0])
    #         y2 = round(match[1].pt[1])
    #         x2 += newColsOffset
    #
    #         count += 1
    #         cv2.line(newIm, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #         #print("x1 = " + str(x1) + " y1 = " + str(y1) + " x2 = " + str(x2) + " y2 = " + str(y2))
    #         print("A match is in: (" + str(x1) + ", "+ str(y1) + ")  - ("+str(round(match[1].pt[0])) +", " +str(y2))
    # print(str(count))
    # cv2.imshow("SIFT", newIm)
    # cv2.waitKey(0)
    ############################## Drawing Matches #####################################



    ################################# RANSAC LOOP #################################
    count = 0
    count1 = 0
    countMax = 0
    maxInliers = 1
    MaxInliersMatchesIndexes = None
    indexListInliers = []
    affineIndexes = None

    numberOfMatches = len(matches)
    numberOfIterations = 1000
    countInliers = 0
    a = 0
    maxInliersList = []
    finalH = None

    match1Mat = np.zeros((2, len(matches)))
    match2Mat = np.zeros((2, len(matches)))
    b = 0
    for loc in locations:
        match1Mat[0][b] = loc[0][0]
        match1Mat[1][b] = loc[0][1]
        match2Mat[0][b] = loc[1][0]
        match2Mat[1][b] = loc[1][1]
        b += 1

    for a in range(numberOfIterations):
        i = int(random.uniform(0, numberOfMatches))
        j = int(random.uniform(0, numberOfMatches))
        k = int(random.uniform(0, numberOfMatches))
        while i == j or i == k or j == k:
            i = int(random.uniform(0, numberOfMatches))
            j = int(random.uniform(0, numberOfMatches))
            k = int(random.uniform(0, numberOfMatches))

        pointSet_Source_Affine = np.float32([[round(locations[i][0][0]), round(locations[i][0][1])],
                                                  [round(locations[j][0][0]), round(locations[j][0][1])],
                                                  [round(locations[k][0][0]), round(locations[k][0][1])]])

        pointSet_Target_Affine = np.float32([[round(locations[i][1][0]), round(locations[i][1][1])],
                                                  [round(locations[j][1][0]), round(locations[j][1][1])],
                                                  [round(locations[k][1][0]), round(locations[k][1][1])]])

        affineTransformation = cv2.getAffineTransform(pointSet_Source_Affine, pointSet_Target_Affine)


        checkAffine = np.zeros((3, 3))
        for i in range(2):
            for j in range(3):
                checkAffine[i][j] = affineTransformation[i][j]
        checkAffine[2][2] = 1
        for loc in locations:
            temp = np.ones((3, 1))
            temp[0][0] = loc[0][0]
            temp[1][0] = loc[0][1]

            HomogenicTemp = np.dot(checkAffine, temp)
            EuclideanTemp = (1/HomogenicTemp.item(2))*HomogenicTemp

            tempLocation = np.ones((3,1))
            tempLocation[0][0] = loc[1][0]
            tempLocation[1][0] = loc[1][1]
            EuclideanDistance = cv2.norm(EuclideanTemp - tempLocation, cv2.NORM_L2)
            if EuclideanDistance < 50:  # Threshold for Inliers
                countInliers += 1

        if countInliers > maxInliers:
            maxInliers = countInliers
            MaxInliersMatchesIndexes = (i, j, k)
            affineIndexes = (i,j,k)
            Affine = affineTransformation
            indexListInliers.clear()
            indexListInliers.append(MaxInliersMatchesIndexes)
            print("max inliers: " + str(maxInliers))

        elif countInliers == maxInliers:
            MaxInliersMatchesIndexes = (i, j, k)
            indexListInliers.append(MaxInliersMatchesIndexes)

        countInliers = 0


    ######################### fitting ############################
    if affineIndexes is not None:
        x = affineIndexes[0]
        y = affineIndexes[1]
        z = affineIndexes[2]

        pointSet_Source_Affine = np.float32([[round(locations[x][0][0]), round(locations[x][0][1])],
                                                  [round(locations[y][0][0]), round(locations[y][0][1])],
                                                  [round(locations[z][0][0]), round(locations[z][0][1])]])

        pointSet_Target_Affine = np.float32([[round(locations[x][1][0]), round(locations[x][1][1])],
                                                  [round(locations[y][1][0]), round(locations[y][1][1])],
                                                  [round(locations[z][1][0]), round(locations[z][1][1])]])

        affineTransformation = cv2.getAffineTransform(pointSet_Source_Affine,
                                                                pointSet_Target_Affine)
        shap = np.shape(img2)
        rows = shap[0]
        cols = shap[1]
        imgOutput = cv2.warpAffine(img1, affineTransformation, (round(cols * 1.5), round(rows * 1.5)))

        grayImage = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2GRAY)
        for i in range(img1.shape[0]):
            for j in range(img2.shape[1]):
                if grayImage[i][j] == 0:
                    imgOutput[i][j] = img2[i][j]
        count += 1
        print("number og images: " + str(count))
        print("max inliers: " + str(maxInliers))
        return imgOutput
