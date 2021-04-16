from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, \
    QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Hough(QWidget):
    def __init__(self):
        super().__init__()
        self.im = None
        self.setWindowTitle("Hough Transform - Finding Triangles")
        self.threshold = None  # minimum number of votes (intersections in Hough grid cell)
        self.min_line_length = None  # minimum number of pixels making up a line
        self.max_line_gap = None  # maximum gap in pixels between connectable line segments
        mainHBox = QHBoxLayout()
        self.image_box = QLabel()
        mainHBox.addWidget(self.image_box)
        buttonsVBox = QVBoxLayout()
        upload_1image002_btn = QPushButton("Upload triangles_1/image002")
        upload_1image002_btn.clicked.connect(lambda: self.upload_image("triangles_1/image002.jpg", 1, 1, 5)) # Threshold, min_line_length, max_line_gap
        upload_1image003_btn = QPushButton("Upload triangles_1/image003")
        upload_1image003_btn.clicked.connect(lambda: self.upload_image("triangles_1/image003.jpg", 20, 40, 40))
        upload_2image001_btn = QPushButton("Upload triangles_2/image001")
        upload_2image001_btn.clicked.connect(lambda: self.upload_image("triangles_2/image001.jpg", 20, 40, 40))
        upload_2image006_btn = QPushButton("Upload triangles_2/image006")
        upload_2image006_btn.clicked.connect(lambda: self.upload_image("triangles_2/image006.jpg", 20, 40, 40))
        find_triangles_btn = QPushButton("Find Tringles")
        find_triangles_btn.clicked.connect(self.find_triangles_using_hough)
        buttonsVBox.addWidget(upload_1image002_btn)
        buttonsVBox.addWidget(upload_1image003_btn)
        buttonsVBox.addWidget(upload_2image001_btn)
        buttonsVBox.addWidget(upload_2image006_btn)
        buttonsVBox.addWidget(find_triangles_btn)
        buttonsWidget = QWidget()
        buttonsWidget.setLayout(buttonsVBox)
        mainHBox.addWidget(buttonsWidget)
        self.setLayout(mainHBox)

    def find_triangles_using_hough(self):
        if self.image_box.pixmap():
            ## canny edge detection
            grayscale = cv2.cvtColor(self.im[1], cv2.COLOR_BGR2GRAY)
            im = cv2.GaussianBlur(self.im[1], (5, 5), cv2.BORDER_DEFAULT)
            #im = cv2.addWeighted(self.im[1], 3, im, -1, 0)
            #im = 255 - cv2.medianBlur(im, 3)
            cv2.imshow("bla2", im)
            low_threshold = 150
            high_threshold = 240
            imgCanny = cv2.Canny(im, low_threshold, high_threshold)
            ''' Gradient Direction '''
            sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
            EdgeGradientDirection = sobely
            rows, cols = np.shape(sobelx)
            for i in range(rows):
                for j in range(cols):
                    if sobelx[i][j] == 0:
                        sobelx[i][j] = 0.00000000000001
                    try:
                        EdgeGradientDirection[i][j] = np.arctan(sobely[i][j] / sobelx[i][j])
                    except:
                        EdgeGradientDirection[i][j] = 0

            cv2.imshow("bla4", imgCanny)
            cv2.imshow("bla8", EdgeGradientDirection)

            # Function to do hough line transform
            H = self.hough_line(imgCanny, EdgeGradientDirection)
            fig = plt.figure(figsize=(10, 10))

            plt.imshow(H, cmap='jet')

            plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
            plt.tight_layout()
            plt.show()

            # Threshold some high values then draw the line
            edge_pixels = np.where(H > 50)
            coordinates = list(zip(edge_pixels[0], edge_pixels[1]))
            line_image = np.copy(self.im[1]) * 0  # creating a blank to draw lines on

            # Use line equation to draw detected line on an original image
            for i in range(0, len(coordinates)):
                a = np.cos(coordinates[i][1])
                b = np.sin(coordinates[i][1])
                x0 = a * coordinates[i][0]
                y0 = b * coordinates[i][0]
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # rho = 1  # distance resolution in pixels of the Hough grid
            # theta = np.pi / 180  # angular resolution in radians of the Hough grid

            # Draw the lines on the  image
            lines_edges = cv2.addWeighted(self.im[1], 0.8, line_image, 1, 0)
            #
            cv2.imshow("bla3", line_image)
            #height, width = imgCanny.shape[:2]
            #edges2 = QImage(imgCanny, width, height, QImage.Format_Grayscale8)
            # pixmap = QPixmap.fromImage(edges2)
            # self.image_box.setPixmap(pixmap)
            cv2.imshow("bla", lines_edges)
        else:
            QMessageBox.critical(self, "Error", "Please upload image first")

    def upload_image(self, path, threshold, min_line_length, max_line_gap):
        self.im = (QPixmap(path), cv2.imread(path))
        self.image_box.setPixmap(self.im[0])
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap


    # ------------------ Do Hough Line Transform ------------------ #
    def hough_line(self, edge, Gradient):
        max_val = -1000000000
        min_val = 100000000

        # create the empty Hough Accumulator with dimensions equal to the size of
        # rhos and thetas
        Gradient_degrees = np.degrees(Gradient)
        for i in range(Gradient.shape[0]):
            for j in range(Gradient.shape[1]):
                if edge[i][j] == 255:
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
        rho_range = round(np.sqrt(edge.shape[0] ** 2 + edge.shape[1] ** 2))
        accumulator = np.zeros((2 * rho_range, len(theta)), dtype=np.uint64)

        # Threshold to get edges pixel location (x,y)
        edge_pixels = np.where(edge == 255)
        coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

        # Calculate rho value for each edge location (x,y) with all the theta range
        for p in range(len(coordinates)):
            for t in range(len(theta)):
                rho = round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t]) + rho_range
                accumulator[rho, t] += 2  # Suppose add 1 only, Just want to get clear result

        return accumulator


if __name__== '__main__':
    import sys
    app = QApplication(sys.argv)
    wnd = Hough()
    wnd.show()
    sys.exit(app.exec_())