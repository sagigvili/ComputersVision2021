import numpy as np
import cv2 as cv
from pylab import *

colors = [(255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (255, 0, 255), (0, 0, 0),
          (255, 255, 255)]
# img1 = cv.imread("Resources/im_courtroom_00086_left.jpg")
# img2 = cv.imread("Resources/im_courtroom_00089_right.jpg")
img1 = cv.imread("Resources/im_family_00084_left.jpg")
img2 = cv.imread("Resources/im_family_00100_right.jpg")
# fig = plt.figure(dpi=200)
# axes = fig.add_subplot(111)
#
# print("After 2 clicks :")
# axes.imshow(img2)
# x = fig.ginput(1, timeout=0)
# print(x)

pts1 = [(122.40322580645162, 51.65645161290308), (225.95161290322582, 303.26935483870955),
        (804.6612903225806, 282.9467741935483),
        (407.88709677419354, 155.20483870967735), (492.08064516129036, 193.9145161290321),
        (71.11290322580646, 112.62419354838698), (901.5533440821101, 391.547740299065),
        (58.508425336964365, 255.11036994799292)]
pts2 = [(575.4806148720278, 36.82515523247025), (126.71490231712855, 305.3091776465244),
        (768.3626381947598, 344.07943359079206),
        (285.6729516886263, 137.62782068756644), (481.4627442071785, 185.12138421929433),
        (705.3609722853248, 102.73459033772542), (719.9907734203722, 512.589454073557),
        (537.3664333395598, 240.24559808095)]

# pts1 = [(207.1561440073931, 28.953483685126344), (394.0275617645035, 231.7514427283541),
#         (615.9373703510721, 191.9350895130607), (735.9173147064895, 68.23895219088251),
#         (402.52171711709934, 88.41257115329779), (208.2179134264676, 401.1036650707354),
#         (407.2996795029346, 445.69798067186406), (407.88709677419354, 445.8225806451613)]
# pts2 = [(265.55346205649005, 7.71809530363646), (339.3464366821672, 103.27734302034071),
#         (568.15774649272, 131.94511733535194), (660.0008012426634, 40.102062585408476),
#         (389.2495993786682, 51.78152619522791), (388.1878299595937, 173.3541246792571),
#         (399.33640885987586, 279.0001818771689), (536.5967741935484, 287.11290322580646)]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]


def compute_Correspond_Epilines(points1, points2, F):
    epi_lines1 = []
    epi_lines2 = []
    for i in range(len(pts1)):
        a = np.ones((3, 1))
        a[0][0] = points1[i][0]
        a[1][0] = points1[i][1]
        eline2 = dot(F, a)
        epi_lines2.append(eline2)

        b = np.ones((3, 1))
        b[0][0] = points2[i][0]
        b[1][0] = points2[i][1]
        eline1 = dot(np.transpose(F), b)
        epi_lines1.append(eline1)
    return epi_lines1, epi_lines2


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    kk, c, e = img1.shape
    # img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    # img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    for r, pt1, pt2, color in zip(lines, pts1, pts2, colors):
        x0, y0 = [0, np.round(-r[2] / r[1])]
        x1, y1 = [np.round(c), np.round(-(r[2] + r[0] * c) / r[1])]
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


lines1, lines2 = compute_Correspond_Epilines(pts1, pts2, F)

results_algeb = []
for pt1, pt2 in zip(pts1, pts2):
    results_algeb.append(np.transpose(np.array([[pt2[0]], [pt2[1]], [1]])) @ F @ np.array([[pt1[0]], [pt1[1]], [1]]))

results_epi_1 = []
for i, line in enumerate(lines1):
    d1 = results_algeb[i] / np.sqrt(np.power(line[0], 2) + np.power(line[1], 2))  # d(x',Fx)
    algeb_res_d2 = (np.transpose(np.array([[pts1[i][0]], [pts1[i][1]], [1]])) @ np.transpose(F) @ np.array([[pts2[i][0]], [pts2[i][1]], [1]]))
    d2 = algeb_res_d2 / np.sqrt(np.power(line[0], 2) + np.power(line[1], 2))  # d(x,F.Tx')
    euclidean_res = np.power(d1, 2) + np.power(d2, 2)
    results_epi_1.append(euclidean_res)

print("The algebraic distance is {}".format(np.average(results_algeb)))

print("The Epipolar distance for left image is {}".format(np.average(results_epi_1)))

im1, im2 = drawlines(img1, img2, lines1, pts1, pts2)
im3, im4 = drawlines(img2, img1, lines2, pts2, pts1)

cv.imshow("Image 1", im1)
cv.imshow("Image 2", im2)
cv.waitKey(0)
