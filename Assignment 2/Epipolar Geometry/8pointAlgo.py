import numpy as np
import cv2 as cv
from pylab import *

colors = [(255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (255, 0, 255), (0, 0, 0)]
img1 = cv.imread("Resources/im_family_00084_left.jpg")
img2 = cv.imread("Resources/im_family_00100_right.jpg")
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k=2)
# pts1 = []
# pts2 = []
# # ratio test as per Lowe's paper
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.8 * n.distance:
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)
#

pts1 = [(122.40322580645162, 51.65645161290308), (225.95161290322582, 303.26935483870955),
        (804.6612903225806, 282.9467741935483), (855.9516129032259, 210.36612903225796),
        (407.88709677419354, 155.20483870967735), (492.08064516129036, 193.9145161290321),
        (71.11290322580646, 112.62419354838698)]
pts2 = [(575.4806148720278, 36.82515523247025), (126.71490231712855, 305.3091776465244),
        (768.3626381947598, 344.07943359079206), (811.0099197334545, 211.29130698167512),
        (285.6729516886263, 137.62782068756644), (481.4627442071785, 185.12138421929433),
        (705.3609722853248, 102.73459033772542)]
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_7POINT)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]


def compute_Correspond_Epilines(points, F):
    lines = []
    for pt in points:
        a = np.ones((3, 1))
        a[0][0] = pt[0]
        a[1][0] = pt[1]
        L = np.dot(F, a)

        lines.append(L)
    return lines


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    kk, c, e = img1.shape
    # img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    # img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    for r, pt1, pt2, color in zip(lines, pts1, pts2, colors):
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 2)
        #img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


lines1 = compute_Correspond_Epilines(pts1, F)

lines2 = compute_Correspond_Epilines(pts2, F)

im1, im2 = drawlines(img1, img2, lines1, pts1, pts2)
im3, im4 = drawlines(img2, img1, lines2, pts2, pts1)

cv.imshow("Sason1", im1)
cv.imshow("Sason2", im2)
cv.waitKey(0)
