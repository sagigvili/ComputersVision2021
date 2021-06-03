import cv2 as cv
from pylab import *
import numpy as np

colors = [(255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (255, 0, 255), (0, 0, 0),
          (255, 255, 255), (128, 255, 0)]

img1 = cv.imread("Resources/im_courtroom_00086_left.jpg")
img2 = cv.imread("Resources/im_courtroom_00089_right.jpg")
# img1 = cv.imread("Resources/im_family_00084_left.jpg")
# img2 = cv.imread("Resources/im_family_00100_right.jpg")


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

# Family Points
# pts1 = [
#         (932.166969559291, 367.3415871573527),
#         (458.20059064061786, 155.0744358624869), (274.04187490534605, 183.182871422081),
#         (121.86862032409508, 51.36400121157067), (268.22633651370586, 124.05823110707263),
#         (219.76351658337117, 204.50651219142821), (629.7589731940027, 228.73792215659557),
#         (59.83621081326669, 254.90784491897625)]
#
# pts2 = [
#         (557.0647432985006, 527.2688929274573),
#         (345.7668484022414, 135.689307890353), (445.60025745873077, 170.5825382401939),
#         (577.4191276692412, 35.855898833863534), (747.0389974254126, 97.88830834469195),
#         (752.8545358170528, 191.90617900954123), (713.1150234741783, 228.73792215659557),
#         (538.6488717249734, 238.43048614266252)]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)

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

print("The Epipolar distance is {}".format(np.average(results_epi)))

im1, im2 = drawlines(img1, img2, lines1, pts1, pts2)
im3, im4 = drawlines(img2, img1, lines2, pts2, pts1)

cv.imwrite("epipolar_im_court_8Point.jpg", np.hstack((im1, im2)))
cv.imshow("Image 1", im1)
cv.imshow("Image 2", im2)
cv.waitKey(0)
