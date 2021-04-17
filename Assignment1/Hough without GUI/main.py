import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("Resources/image003.jpg")

''' Edge Detection by Canny '''
shapes_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgBlur = cv2.GaussianBlur(shapes_grayscale, (5, 5), cv2.BORDER_DEFAULT)
low_threshold = 150
high_threshold = 240
imgCanny = cv2.Canny(imgBlur, low_threshold, high_threshold)

''' Gradient Direction '''
dx = cv2.Sobel(imgCanny, cv2.CV_64F, 1, 0, ksize=3)
dy = cv2.Sobel(imgCanny, cv2.CV_64F, 0, 1, ksize=3)
EdgeGradientDirection = dy
rows, cols = np.shape(dx)
for i in range(rows):
    for j in range(cols):
        if dx[i][j] == 0:
            dx[i][j] = 0.00000000000001
        EdgeGradientDirection[i][j] = np.arctan(dy[i][j] / dx[i][j])

#
# dx, dy = np.gradient(imgCanny)
# rows, cols = dx.shape
# EdgeGradientDirection = np.zeros(dx.shape)
# for i in range(rows):
#     for j in range(cols):
#         if dx[i][j] == 0:
#             dx[i][j] = 0.000000000001
#         EdgeGradientDirection[i][j] = np.arctan(dy[i][j] / dx[i][j])

# img_diagonal = round(np.sqrt(rows ** 2 + cols ** 2))
# d = np.arange(0, img_diagonal, 1)
#
# max_val = -100000000000000
# min_val = 100000000000000
# for i in range(rows):
#     for j in range(cols):
#         if EdgeGradientDirection[i][j] > max_val:
#             max_val = EdgeGradientDirection[i][j]
#         if EdgeGradientDirection[i][j] < min_val:
#             min_val = EdgeGradientDirection[i][j]
#
# min_val = round(np.rad2deg(min_val))
# max_val = round(np.rad2deg(max_val))
# thetas = np.arange(min_val, max_val + 1, 1)
#
# H = np.zeros((len(d), len(thetas)), dtype=np.uint64)
#
# for i in range(rows):
#     for j in range(cols):
#         if imgCanny[i][j] == 255:
#             raw = j * np.cos(EdgeGradientDirection[i][j]) + i * np.sin(EdgeGradientDirection[i][j])
#             EdgeGradientDirectionInDegrees = round(np.rad2deg(EdgeGradientDirection[i][j]))
#             H[round(raw), EdgeGradientDirectionInDegrees] += 1


max_val = -1000000000
min_val = 100000000

''' create the empty Hough Accumulator with dimensions equal to the size of rhos and thetas '''
Gradient_degrees = np.rad2deg(EdgeGradientDirection)
for i in range(EdgeGradientDirection.shape[0]):
    for j in range(EdgeGradientDirection.shape[1]):
        if imgCanny[i][j] == 255:
            if Gradient_degrees[i][j] > max_val:
                max_val = Gradient_degrees[i][j]
            if Gradient_degrees[i][j] < min_val:
                min_val = Gradient_degrees[i][j]

# Theta 0 - 180 degree
# Calculate 'cos' and 'sin' value ahead to improve running time
theta = np.arange(round(min_val), round(max_val) + 1, 1)
cos = np.cos(np.deg2rad(theta))
sin = np.sin(np.deg2rad(theta))

# Generate a accumulator matrix to store the values
rho_range = round(np.sqrt(imgCanny.shape[0] ** 2 + imgCanny.shape[1] ** 2))
accumulator = np.zeros((rho_range, len(theta)), dtype=np.uint64)


# Threshold to get edges pixel location (x,y)
edge_pixels = np.where(imgCanny == 255)
coordinates = list(zip(edge_pixels[0], edge_pixels[1]))


# Calculate rho value for each edge location (x,y) with all the theta range
for p in range(len(coordinates)):
    for t in range(len(theta)):
        rho = round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t])# + rho_range
        accumulator[rho, t] += 1  # Suppose add 1 only, Just want to get clear result


# Threshold some high values then draw the line
edge_pixels = np.where(accumulator > 50)
coordinates = list(zip(edge_pixels[0], edge_pixels[1]))
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Use line equation to draw detected line on an original image
for i in range(len(coordinates)):
    a = np.cos(coordinates[i][1])
    b = np.sin(coordinates[i][0])
    x0 = a * coordinates[i][1]
    y0 = b * coordinates[i][0]
    scale = 10
    x1 = int(x0 + scale * (-b))
    y1 = int(y0 + scale * (a))
    x2 = int(x0 - scale * (-b))
    y2 = int(y0 - scale * (a))

    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

# rho = 1  # distance resolution in pixels of the Hough grid
# theta = np.pi / 180  # angular resolution in radians of the Hough grid

# Draw the lines on the  image
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
#
cv2.imshow("check 1",lines_edges)
fig = plt.figure(figsize=(10, 10))
plt.imshow(accumulator, cmap='jet')
plt.xlabel('Theta Direction'), plt.ylabel('D Direction')
plt.tight_layout()
plt.show()
cv2.waitKey(0)
