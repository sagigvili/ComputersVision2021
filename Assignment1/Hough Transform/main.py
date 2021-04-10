from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, \
    QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import skimage.exposure as exposure


class Hough(QWidget):
    def __init__(self):
        super().__init__()
        self.im = None
        self.setWindowTitle("Hough Transform - Finding Triangles")
        mainHBox = QHBoxLayout()
        self.image_box = QLabel()
        mainHBox.addWidget(self.image_box)
        buttonsVBox = QVBoxLayout()
        upload_image_btn = QPushButton("Upload Image")
        upload_image_btn.clicked.connect(self.upload_image)
        find_triangles_btn = QPushButton("Find Tringles")
        find_triangles_btn.clicked.connect(self.find_triangles_using_hough)
        buttonsVBox.addWidget(upload_image_btn)
        buttonsVBox.addWidget(find_triangles_btn)
        buttonsWidget = QWidget()
        buttonsWidget.setLayout(buttonsVBox)
        mainHBox.addWidget(buttonsWidget)
        self.setLayout(mainHBox)

    def find_triangles_using_hough(self):
        if self.image_box.pixmap():
            ## canny edge detection

            low_threshold = 350
            high_threshold = 450
            imgCanny = cv2.Canny(self.im[1], low_threshold, high_threshold)

            sobelx = cv2.Sobel(imgCanny, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(imgCanny, cv2.CV_64F, 0, 1, ksize=3)

            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 15  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 60  # minimum number of pixels making up a line
            max_line_gap = 80  # maximum gap in pixels between connectable line segments
            line_image = np.copy(self.im[1]) * 0  # creating a blank to draw lines on

            # Run Hough on edge detected image
            # Output "lines" is an array containing endpoints of detected line segments
            lines = cv2.HoughLinesP(imgCanny, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)

            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # Draw the lines on the  image
            lines_edges = cv2.addWeighted(self.im[1], 0.8, line_image, 1, 0)

            # # apply sobel x derivative
            # # normalize to range 0 to 255
            sobelx_norm8 = exposure.rescale_intensity(sobelx, in_range='image', out_range=(0, 255)).astype(np.uint8)
            # sobely_norm8 = exposure.rescale_intensity(sobely, in_range='image', out_range=(0, 255)).astype(np.uint8)

            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 9, 5, 0.04)

            line_image[dst > 0.01 * dst.max()] = [0, 0, 255]

            height, width = line_image.shape[:2]
            edges2 = QImage(line_image, width, height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(edges2)
            self.image_box.setPixmap(pixmap)
            cv2.imshow("bla", line_image)
        else:
            QMessageBox.critical(self, "Error", "Please upload image first")

    def upload_image(self):
        image_path = QFileDialog.getOpenFileName(self, caption="File to Import",
                                           directory=".", filter="Jpeg Image (*.jpg)")
        if not image_path[0] == "":
            self.im = (QPixmap(image_path[0]), cv2.imread(image_path[0]))
            self.image_box.setPixmap(self.im[0])



if __name__== '__main__':
    import sys
    app = QApplication(sys.argv)
    wnd = Hough()
    wnd.show()
    sys.exit(app.exec_())