from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, \
    QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from scipy import signal


class Hough(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hough Transform - Finding Triangles")
        self.im = None
        self.side_len = None
        self.threshold = None  # minimum number of votes (intersections in Hough grid cell)
        self.canny_low = None
        self.canny_high = None
        self.stride = None
        mainHBox = QHBoxLayout()
        self.image_box = QLabel()
        mainHBox.addWidget(self.image_box)
        buttonsVBox = QVBoxLayout()
        # Low canny threshold, high canny threshold, triangle side length, votes threshold, stride for local maxima
        upload_1image002_btn = QPushButton("Upload triangles_1/image002")
        upload_1image002_btn.clicked.connect(lambda: self.upload_image("triangles_1/image002.jpg", 150, 240, 10, 0, 22))
        upload_1image003_btn = QPushButton("Upload triangles_1/image003")
        upload_1image003_btn.clicked.connect(lambda: self.upload_image("triangles_1/image003.jpg", 150, 240, 98.75, 3, 315))
        upload_2image001_btn = QPushButton("Upload triangles_2/image001")
        upload_2image001_btn.clicked.connect(lambda: self.upload_image("triangles_2/image001.jpg", 1, 10, 30, 10, 60))
        upload_2image006_btn = QPushButton("Upload triangles_2/image006")
        upload_2image006_btn.clicked.connect(lambda: self.upload_image("triangles_2/image006.jpg", 10, 80, 110, 5, 135))
        # upload_test1_btn = QPushButton("Upload Test 1")
        # upload_test1_btn.clicked.connect(lambda: self.upload_image("test_1.jpg", 100, 200, 170, 168, 1))
        # upload_test2_btn = QPushButton("Upload Test 2")
        # upload_test2_btn.clicked.connect(lambda: self.upload_image("test_2.jpg", 100, 200, 170, 20, 1))
        # upload_test3_btn = QPushButton("Upload Test 3")
        # upload_test3_btn.clicked.connect(lambda: self.upload_image("test_3.jpg", 100, 200, 170, 174, 1))
        find_triangles_btn = QPushButton("Find Tringles")
        find_triangles_btn.clicked.connect(self.find_triangles_using_hough)
        buttonsVBox.addWidget(upload_1image002_btn)
        buttonsVBox.addWidget(upload_1image003_btn)
        buttonsVBox.addWidget(upload_2image001_btn)
        buttonsVBox.addWidget(upload_2image006_btn)
        # buttonsVBox.addWidget(upload_test1_btn)
        # buttonsVBox.addWidget(upload_test2_btn)
        # buttonsVBox.addWidget(upload_test3_btn)
        buttonsVBox.addWidget(find_triangles_btn)
        buttonsWidget = QWidget()
        buttonsWidget.setLayout(buttonsVBox)
        mainHBox.addWidget(buttonsWidget)
        self.setLayout(mainHBox)

    def find_triangles_using_hough(self):
        if self.image_box.pixmap():
            ## canny edge detection
            grayscale = cv2.cvtColor(self.im[1], cv2.COLOR_BGR2GRAY)
            im = cv2.GaussianBlur(grayscale, (5, 5), cv2.BORDER_DEFAULT)

            imgCanny = cv2.Canny(im, self.canny_low, self.canny_high)

            ''' Gradient Direction '''
            dx = signal.convolve2d(grayscale, [[-1, 1]], 'same')
            dy = signal.convolve2d(grayscale, [[-1], [1]], 'same')
            EdgeGradientDirection = np.zeros(np.shape(dx))
            rows, cols = EdgeGradientDirection.shape
            for i in range(rows):
                for j in range(cols):
                    EdgeGradientDirection[i][j] = np.arctan2(dy[i][j], dx[i][j])

            cv2.imshow("bla4", imgCanny)
            # cv2.imshow("bla8", EdgeGradientDirection)

            # Function to do hough line transform
            H = self.hough_line(imgCanny, EdgeGradientDirection)
            # arr = np.zeros((imgCanny.shape[0] + 1, imgCanny.shape[1] + 1, 121))
            # for key in H:
            #     arr[key] = H[key]
            print("\n\n\n\n")
            H_peaks = {}
            stride = self.stride
            stride_x = []
            stride_y = []
            temp1 = 0
            temp2 = 0
            # Image shape contains size of x in shape[1], size of y in shape[0]
            while temp1 < imgCanny.shape[1] or temp2 < imgCanny.shape[0]:
                stride_x.append(temp1)
                temp1 += stride
                stride_y.append(temp2)
                temp2 += stride

            for i in stride_x:
                for j in stride_y:
                    curr_max = (0, self.threshold)
                    for x in range(0, stride):
                        for y in range(0, stride):
                            if i + x < imgCanny.shape[0] and j + y < imgCanny.shape[1]:
                                for k in range(121):
                                    if (i + x, j + y, k) in H.keys() and (i + x, j + y, k) not in H_peaks.keys():
                                        if H[(i + x, j + y, k)] > curr_max[1]:
                                            curr_max = ((i + x, j + y, k), H[(i + x, j + y, k)])
                                            # print("Coordinate ({},{},{}), Value is {}".format(i + x, j + y, k,
                                            # H[(i + x, j + y, k)]))
                    if curr_max[0]:
                        # print("The peak is {}, its votes are {}".format(curr_max[0], curr_max[1]))
                        H_peaks[curr_max[0]] = curr_max[1]

            # # Threshold some high values then draw the line
            r = (self.side_len * np.sqrt(3)) / 6
            line_image = np.copy(self.im[1]) * 0  # creating a blank to draw lines on
            for key in H_peaks:
                if H_peaks[key] > self.threshold:
                    # Calculate ABC of the triangle out of centeroid
                    # Calculate three vertices of the triangle
                    point_A = (round(key[1] + (2 * r * np.cos(np.deg2rad(key[2])))),
                               round(key[0] + (2 * r * np.sin(np.deg2rad(key[2])))))  # Top vertex
                    point_B = (round(key[1] + (2 * r * np.cos(np.deg2rad(key[2] + 120)))),
                               round(key[0] + (2 * r * np.sin(np.deg2rad(key[2] + 120)))))  # Top vertex
                    point_C = (round(key[1] + (2 * r * np.cos(np.deg2rad(key[2] + 240)))),
                               round(key[0] + (2 * r * np.sin(np.deg2rad(key[2] + 240)))))  # Top vertex
                    cv2.line(line_image, point_A, point_B, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.line(line_image, point_B, point_C, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.line(line_image, point_C, point_A, (0, 0, 255), 1, cv2.LINE_AA)
                    if 0 <= key[0] < imgCanny.shape[0] and 0 <= key[1] < imgCanny.shape[1]:
                        line_image[key[0]][key[1]] = 255

            #print(max(H_peaks.values()))

            # Draw the lines on the  image
            lines_edges = cv2.addWeighted(self.im[1], 0.8, line_image, 1, 0)
            height, width, bytesPerComponent = lines_edges.shape
            bytesPerLine = 3 * width
            cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB, lines_edges)
            QImg = QImage(lines_edges.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(QImg)
            self.image_box.setPixmap(pixmap)
        else:
            QMessageBox.critical(self, "Error", "Please upload image first")

    def hough_line(self, canny_image, gradients_image):
        a = self.side_len

        # Generate a accumulator matrix to store the values
        accumulator = {}
        edge_pixels = np.where(canny_image == 255)
        coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

        for i in range(len(coordinates)):
            t = gradients_image[coordinates[i][0]][coordinates[i][1]]
            for k in range(-round(a / 2), round(a / 2)):
                t_temp = np.deg2rad(np.rad2deg(t) + 90)
                x_center = coordinates[i][1] + (k * np.cos(t_temp))
                y_center = coordinates[i][0] + (k * np.sin(t_temp))
                r = (a * np.sqrt(3)) / 6
                x_center += (r * np.cos(t))
                y_center += (r * np.sin(t))

                if 0 <= x_center < canny_image.shape[1] and 0 <= y_center < canny_image.shape[0]:
                    #print(round(np.rad2deg(t) % 120))
                    #print("\n")
                    point = (round(y_center), round(x_center), round(np.rad2deg(t) % 120))
                    if point not in accumulator.keys():
                        accumulator[point] = 0
                    accumulator[point] += 1

        return accumulator

    def upload_image(self, path, canny_low, canny_high, side_len, threshold, stride):
        self.im = (QPixmap(path), cv2.imread(path))
        self.image_box.setPixmap(self.im[0])
        self.threshold = threshold
        self.side_len = side_len
        self.canny_high = canny_high
        self.canny_low = canny_low
        self.stride = stride


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    wnd = Hough()
    wnd.show()
    sys.exit(app.exec_())
