import numpy as np
import cv2
import cv_io

depthMapAlley = cv_io.sintel_io.depth_read('alley_2.dpt')
# cv_io.sintel_io.depth_write('test.dpt', image)     Creating depth map
IntrinsicAlley, ExtrinsicAlley = cv_io.sintel_io.cam_read('alley_2.cam')
img1 = cv2.imread("alley_2.png")
r, c, e = np.shape(img1)
inverseIntrinsic = np.linalg.inv(IntrinsicAlley)
cam_points = np.float64(np.zeros((r * c, 3)))
i = 0
# Loop through each pixel in the image
for v in range(r):
    for u in range(c):
        #Apply equation in fig 5
        x = (u - IntrinsicAlley[0][2]) * depthMapAlley[v, u] / IntrinsicAlley[0][0]
        y = (v - IntrinsicAlley[1][2]) * depthMapAlley[v, u] / IntrinsicAlley[1][1]
        z = depthMapAlley[v, u]
        cam_points[i] = (x, y, z)
        i = i + 1

backprojectionMatrix = np.float32(np.zeros((r * c, 2)))
homogenouos3DCoords = np.float32(np.ones((r * c, 4)))
for i in range(r):
    homogenouos3DCoords[i][0] = cam_points[i][0]
    homogenouos3DCoords[i][1] = cam_points[i][1]
    homogenouos3DCoords[i][2] = cam_points[i][2]

newExtrinsic = np.zeros((3,4))
newExtrinsic[0][0] = 1
newExtrinsic[1][1] = 1
newExtrinsic[2][2] = 1
temp = np.ones((1, 4))
#ExtrinsicAlley = np.vstack((ExtrinsicAlley, temp))
cam_mat = np.matmul(IntrinsicAlley, ExtrinsicAlley)
img_cords_Homogeuous = np.matmul(cam_mat, np.transpose(homogenouos3DCoords))

# homogBackProj = np.matmul(np.matmul(IntrinsicAlley, ExtrinsicAlley), np.transpose(homogenouos3DCoords))
#


#homogBackProj = np.transpose(homogBackProj)
k = 0
for i in range(backprojectionMatrix.shape[1]):
    backprojectionMatrix[0][i] = int(np.round(img_cords_Homogeuous[0][i] / img_cords_Homogeuous[2][i]))
    backprojectionMatrix[1][i] = int(np.round(img_cords_Homogeuous[1][i] / img_cords_Homogeuous[2][i]))
    k = k+1
#



'''
THAT'S WORKING
BUT I NEED TO TRY IT WITH THE EXTRINSIC MATRIX
'''
for i in range(r*c):
    u1 = int(np.round((cam_points[i][0] * IntrinsicAlley[0][0] / cam_points[i][2]) + IntrinsicAlley[0][2]))
    v1 = int(np.round((cam_points[i][1] * IntrinsicAlley[1][1] / cam_points[i][2]) + IntrinsicAlley[1][2]))
    backprojectionMatrix[i] = (u1, v1)








newIm = np.zeros(np.shape(img1))
#backprojectionMatrix = np.transpose(backprojectionMatrix)
k = 0
for i in range(r):
    for j in range(c):
        m, n = backprojectionMatrix[k]
        if m>= c or n >= r or m<0 or n < 0:
            pass
        else:
            print("i = " + str(i) + " , j = "+str(j))
            print("m = " + str(m) + " , n = " + str(n))
            newIm[i][j] = img1[int(n)][int(m)]
        k = k+1
        # newIm[i][j] = img1[int(m)][int(n)]
        # newIm[i][j] = img1[int(m)][int(n)]



#cv_io.imshow(depthMapAlley)
cv2.imshow("origin", img1)
cv2.imshow("2D-3D-2D", newIm / 256)
cv2.waitKey(0)
