import numpy as np
import cv2
import cv_io


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(thetaX=0, thetaY=0, thetaZ=0):
    thetaX = np.deg2rad(thetaX)
    thetaY = np.deg2rad(thetaY)
    thetaZ = np.deg2rad(thetaZ)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(thetaX), -np.sin(thetaX)],
                   [0, np.sin(thetaX), np.cos(thetaX)]
                   ])

    Ry = np.array([[np.cos(thetaY), 0, np.sin(thetaY)],
                   [0, 1, 0],
                   [-np.sin(thetaY), 0, np.cos(thetaY)]
                   ])

    Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],
                   [np.sin(thetaZ), np.cos(thetaZ), 0],
                   [0, 0, 1]
                   ])

    R = np.dot(Rz, np.dot(Ry, Rx))

    return R


depthMapAlley = cv_io.sintel_io.depth_read('alley_2.dpt')
# cv_io.sintel_io.depth_write('test.dpt', image)     Creating depth map
IntrinsicAlley, ExtrinsicAlley = cv_io.sintel_io.cam_read('alley_2.cam')
img1 = cv2.imread("alley_2.png")
r, c, e = np.shape(img1)
inverseIntrinsic = np.linalg.inv(IntrinsicAlley)
D3_points = np.float64(np.zeros((r * c, 3)))
D2_points = np.float64(np.ones((r * c, 3)))

i = 0
# Loop through each pixel in the image
for v in range(r):
    for u in range(c):
        D2_points[i] = (v, u, 1)
        i = i + 1

m1 = inverseIntrinsic @ np.transpose(D2_points)
m1 = np.transpose(m1)
D3_points[:, 0] = m1[:, 0] * np.matrix.flatten(depthMapAlley)
D3_points[:, 1] = m1[:, 1] * np.matrix.flatten(depthMapAlley)
D3_points[:, 2] = m1[:, 2] * np.matrix.flatten(depthMapAlley)

# backprojectionMatrix = np.float32(np.zeros((2, r * c)))
homogenouos3DCoords = np.float32(np.ones((r * c, 4)))
for i in range(r * c):
    homogenouos3DCoords[i][0] = D3_points[i][0]
    homogenouos3DCoords[i][1] = D3_points[i][1]
    homogenouos3DCoords[i][2] = D3_points[i][2]


angle = 0
imNumber = 0
for angle in np.arange(0, 5, 0.05):
    backprojectionMatrix = np.float32(np.zeros((2, r * c)))
    T = np.array([[0], [angle], [0]])
    R = eulerAnglesToRotationMatrix(0, 0, 0)
    newExtrinsic = np.hstack((R, T))
    cam_mat = IntrinsicAlley @ newExtrinsic
    img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)

    for i in range(r * c):
        backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
        backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))

    newIm = np.zeros(np.shape(img1))
    backprojectionMatrix = np.transpose(backprojectionMatrix)
    k = 0
    for i in range(r):
        for j in range(c):
            m, n = backprojectionMatrix[k]
            if n >= c or m >= r or n < 0 or m < 0:
                pass
            else:
                newIm[i][j] = img1[int(m)][int(n)]
            k = k + 1

    # cv_io.imshow(depthMapAlley)
    #cv2.imshow("origin", img1)
    cv2.imwrite("im  test " + str(imNumber)+".jpg", newIm)
    print("image  number: " + str(imNumber))
    imNumber = imNumber + 1


# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(15 - angle, 0, 0)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     #cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(-angle, 0, 0)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     #cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(angle - 15, 0, 0)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     #cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber)+".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(0, angle, 0)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     # cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber) + ".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(0, 15 - angle, 0)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     # cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber) + ".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(0, -angle, 0)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     # cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber) + ".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(0, angle - 15, 0)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     # cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber) + ".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(0, 0, angle)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     # cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber) + ".jpg", newIm)
#     # cv2.waitKey(0)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
#
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(0, 0, 15 - angle)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     # cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber) + ".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(0, 0, -angle)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     # cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber) + ".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1
#
# for angle in np.arange(0, 15, 0.5):
#     backprojectionMatrix = np.float32(np.zeros((2, r * c)))
#     T = np.array([[0], [0], [0]])
#     R = eulerAnglesToRotationMatrix(0, 0, angle - 15)
#     newExtrinsic = np.hstack((R, T))
#     cam_mat = IntrinsicAlley @ newExtrinsic
#     img_cords_Homogenuous = cam_mat @ np.transpose(homogenouos3DCoords)
#
#     for i in range(r * c):
#         backprojectionMatrix[0][i] = int(np.round(img_cords_Homogenuous[0][i] / img_cords_Homogenuous[2][i]))
#         backprojectionMatrix[1][i] = int(np.round(img_cords_Homogenuous[1][i] / img_cords_Homogenuous[2][i]))
#
#     newIm = np.zeros(np.shape(img1))
#     backprojectionMatrix = np.transpose(backprojectionMatrix)
#     k = 0
#     for i in range(r):
#         for j in range(c):
#             m, n = backprojectionMatrix[k]
#             if n >= c or m >= r or n < 0 or m < 0:
#                 pass
#             else:
#                 newIm[i][j] = img1[int(m)][int(n)]
#             k = k + 1
#
#     # cv_io.imshow(depthMapAlley)
#     # cv2.imshow("origin", img1)
#     cv2.imwrite("im " + str(imNumber) + ".jpg", newIm)
#     print("image number: " + str(imNumber))
#     imNumber = imNumber + 1