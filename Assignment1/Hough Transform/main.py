from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, \
    QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np


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
            imgCanny = cv2.Canny(self.im[1], 200, 300)
            '''
            lines = cv2.HoughLines(imgCanny, 1, np.pi / 360, 200)
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(self.im[1], (x1,y1), (x2,y2), (255,0,0), 2)
            ##cv2.filter
            #imgEdegeGradient = cv2.Laplacian(imgCanny, cv2.CV_8U)
            cv2.imshow("Laplacian Image", self.im[1])
            '''
            cv2.imshow("Canny Image", imgCanny)
            cv2.waitKey(0)
            '''
            image = QImage(imgCanny.data, imgCanny.shape[1], imgCanny.shape[0], QImage.Format_RGB888).rgbSwapped()
            self.image_box.setPixmap(QPixmap.fromImage(image))'''

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