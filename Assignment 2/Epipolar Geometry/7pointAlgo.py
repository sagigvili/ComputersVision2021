import numpy as np
import cv2 as cv
from pylab import *

colors = [(255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (255, 0, 255), (0, 0, 0),
          (255, 255, 255), (128, 255, 0)]
img1 = cv.imread("Resources/im_courtroom_00086_left.jpg")
img2 = cv.imread("Resources/im_courtroom_00089_right.jpg")
# img1 = cv.imread("Resources/im_family_00084_left.jpg")
# img2 = cv.imread("Resources/im_family_00100_right.jpg")
# fig = plt.figure(dpi=200)
# axes = fig.add_subplot(111)
#
# print("After 2 clicks :")
# axes.imshow(img2)
# x = fig.ginput(11, timeout=0)
# print(x)

# pts1 = np.int32([(280, 130), (522, 222), (747, 80),
#                                 (742, 200), (427, 292), (613, 379), (215, 311),
#                                 (619, 527)])
# pts2 = np.int32([(313, 91), (486, 160), (669, 48),
#                                  (667, 135), (457, 214), (624, 255), (332, 232),
#                                  (632, 342)])

# Court Points
pts1 = [(208.53225806451613, 27.758064516129025), (395.3064516129032, 231.9516129032258), (506.5967741935484, 155.5),
        (614.016129032258, 194.20967741935488), (736.9193548387096, 73.24193548387098), (850.1451612903226, 215.5),
        (793.0483870967741, 337.43548387096774),
        (404.01612903225805, 443.88709677419354), (28.532258064516128, 310.33870967741933),
        (424.3387096774194, 280.33870967741933)]

pts2 = [(265.6290322580645, 15.177419354838662), (389.5, 171.95161290322574), (476.5967741935484, 106.14516129032256),
        (566.5967741935484, 134.20967741935488), (661.4354838709677, 45.17741935483866),
        (740.7903225806451, 144.85483870967744),
        (645.9516129032259, 285.1774193548387), (535.6290322580645, 284.2096774193548),
        (284.98387096774195, 225.17741935483872), (452.4032258064516, 208.72580645161293)]

# Park Points
# pts1 = [(122.40322580645162, 51.65645161290308), (225.95161290322582, 303.26935483870955),
#         (804.6612903225806, 282.9467741935483),
#         (407.88709677419354, 155.20483870967735), (492.08064516129036, 193.9145161290321),
#         (71.11290322580646, 112.62419354838698), (901.5533440821101, 391.547740299065),
#         (58.508425336964365, 255.11036994799292)]
# pts2 = [(575.4806148720278, 36.82515523247025), (126.71490231712855, 305.3091776465244),
#         (768.3626381947598, 344.07943359079206),
#         (285.6729516886263, 137.62782068756644), (481.4627442071785, 185.12138421929433),
#         (705.3609722853248, 102.73459033772542), (719.9907734203722, 512.589454073557),
#         (537.3664333395598, 240.24559808095)]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_7POINT)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]


def compute_Correspond_Epilines(points1, points2, F):
    epi_lines1 = []
    epi_lines2 = []
    for i in range(len(points1)):
        a = np.ones((3, 1))
        a[0][0] = points1[i][0]
        a[1][0] = points1[i][1]
        eline2 = F @ a
        epi_lines2.append(eline2)

        b = np.ones((3, 1))
        b[0][0] = points2[i][0]
        b[1][0] = points2[i][1]
        eline1 = np.transpose(F) @ b
        epi_lines1.append(eline1)
    return epi_lines1, epi_lines2


def drawlines(img1, img2, lines, pts1, pts2):
    kk, c, e = img1.shape
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
    results_algeb.append(np.transpose(np.array([pt2[0], pt2[1], 1])) @ F @ np.array([pt1[0], pt1[1], 1]))

# errors2.append(
#     power(dis1/sqrt(power(fx[0], 2) + power(fx[1], 2)), 2)
#     +
#     power(dis2/sqrt(power(fx[0], 2) + power(fx[1], 2)), 2))

results_epi = []
for i in range(len(pts1)):
    x_homogen = np.ones((3, 1))
    x_homogen[0] = pts1[i][0]
    x_homogen[1] = pts1[i][1]

    x_tag_homogen = np.ones((3, 1))
    x_tag_homogen[0] = pts2[i][0]
    x_tag_homogen[1] = pts2[i][1]

    Fx = F @ x_homogen
    alegb_dis1 = abs(np.transpose(x_tag_homogen) @ Fx)

    F_t_x_tag = np.transpose(F) @ x_tag_homogen
    alegb_dis2 = abs(np.transpose(x_homogen) @ F_t_x_tag)

    final_d = ((alegb_dis1 ** 2) + (alegb_dis2 ** 2)) / ((Fx[0] ** 2) + (Fx[1] ** 2))
    results_epi.append(final_d)

print("The algebraic distance is {}".format(np.average(np.abs(results_algeb))))

print("The Epipolar distance for right image is {}".format(np.average(results_epi)))

im1, im2 = drawlines(img1, img2, lines1, pts1, pts2)
im3, im4 = drawlines(img2, img1, lines2, pts2, pts1)

cv.imshow("Image 1", im1)
cv.imshow("Image 2", im2)
cv.waitKey(0)
