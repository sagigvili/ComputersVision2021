import numpy as np
import cv2
import cv_io

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(thetaX=0, thetaY=0, thetaZ=0):
    thetaX = np.deg2rad(thetaX)
    thetaY = np.deg2rad(thetaY)
    thetaZ = np.deg2rad(thetaZ)
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(thetaX), -np.sin(thetaX), 0],
                   [0, np.sin(thetaX), np.cos(thetaX), 0],
                   [0, 0, 0, 1]
                   ])

    Ry = np.array([[np.cos(thetaY), 0, np.sin(thetaY), 0],
                   [0, 1, 0, 0],
                   [-np.sin(thetaY), 0, np.cos(thetaY), 0],
                   [0, 0, 0, 1]
                   ])

    Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0, 0],
                   [np.sin(thetaZ), np.cos(thetaZ), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]
                   ])

    R = np.dot(Rz, np.dot(Ry, Rx))

    return R

depthMapAlley = cv_io.sintel_io.depth_read('alley_2.dpt')
# cv_io.sintel_io.depth_write('test.dpt', image)     Creating depth map
IntrinsicAlley, ExtrinsicAlley = cv_io.sintel_io.cam_read('alley_2.cam')
img1 = cv2.imread("alley_2.png")
r, c, e = np.shape(img1)
inverseIntrinsic = np.linalg.inv(IntrinsicAlley)
cam_points = np.float64(np.zeros((r * c, 3)))
i = 0
imNumber = 0
for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(angel, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(10 - angel, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(-angel, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(angel - 10, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

################################################################

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, angel, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 10 - angel, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, -angel, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, angel - 10, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

################################################################

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, angel)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, 10 - angel)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, -angel)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 10, 0.5):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, angel - 10)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

################################################################

################################################################

################################################################

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = angel
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = 0.15 - angel
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = -angel
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = angel - 0.15
    T[1][3] = 0
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

#####################################################################

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = angel
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0.15 - angel
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = -angel
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = angel - 0.15
    T[2][3] = 0

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

#####################################################################

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = angel

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = 0.15 - angel

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = -angel

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

############

for angel in np.arange(0, 0.15, 0.01):
    T = np.identity(4)
    T[0][3] = 0
    T[1][3] = 0
    T[2][3] = angel - 0.15

    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = R @ T
    newExtrinsic = newExtrinsic[0:3, 0:4]
    P = IntrinsicAlley @ newExtrinsic
    boolDoubleMap = np.zeros((r, c))
    newIm = np.zeros(np.shape(img1))
    # Loop through each pixel in the image
    for v in range(r):
        for u in range(c):
            x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
            y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
            z = depthMapAlley[v, u]
            D3_point_homogenic = (x, y, z, 1)
            transformed_D3_point_homogenic = P @ D3_point_homogenic
            if transformed_D3_point_homogenic[2] > 0:
                new_x = int(transformed_D3_point_homogenic[0]/transformed_D3_point_homogenic[2])
                new_y = int(transformed_D3_point_homogenic[1]/transformed_D3_point_homogenic[2])
                if 0 <= new_x < newIm.shape[1] and 0 <= new_y < newIm.shape[0]:
                    if boolDoubleMap[new_y][new_x] > 0:
                        if boolDoubleMap[new_y][new_x] > z:
                            newIm[new_y][new_x] = img1[v][u]
                            boolDoubleMap[new_y][new_x] = z
                    else:
                        newIm[new_y][new_x] = img1[v][u]
                        boolDoubleMap[new_y][new_x] = z


    #cv_io.imshow(depthMapAlley)
    cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
    print("image number: " + str(imNumber))
    imNumber = imNumber + 1

#####################################################################