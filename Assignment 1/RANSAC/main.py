from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from AffineRANSAC import *
from perspectiveRansac import *
from Warping import *


class RANSAC(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RANSAC")
        mainHBox = QHBoxLayout()

        top_img_hbox = QHBoxLayout()
        self.orig_img1 = QLabel()
        self.orig_img2 = QLabel()
        top_img_hbox.addWidget(self.orig_img1)
        top_img_hbox.addWidget(self.orig_img2)
        top_img_widget = QWidget()
        top_img_widget.setLayout(top_img_hbox)

        imgsVBox = QVBoxLayout()
        imgsVBox.addWidget(top_img_widget)
        self.image_box = QLabel()
        imgsVBox.addWidget(self.image_box, 0, Qt.AlignCenter)

        imgsWidget = QWidget()
        imgsWidget.setLayout(imgsVBox)

        mainHBox.addWidget(imgsWidget)
        buttonsVBox = QVBoxLayout()
        warping_btn = QPushButton("Show Warping")
        warping_btn.clicked.connect(self.run_warping)
        affine_ransac_btn1 = QPushButton("Affine RANSAC pair1")
        affine_ransac_btn1.clicked.connect(lambda: self.run_affine_runsac("Resources/pair1_imageA.jpg", "Resources/pair1_imageB.jpg"))
        affine_ransac_btn2 = QPushButton("Affine RANSAC pair2")
        affine_ransac_btn2.clicked.connect(lambda: self.run_affine_runsac("Resources/pair2_imageA.jpg", "Resources/pair2_imageB.jpg"))
        affine_ransac_btn3 = QPushButton("Affine RANSAC pair3")
        affine_ransac_btn3.clicked.connect(lambda: self.run_affine_runsac("Resources/pair3_imageA.jpg", "Resources/pair3_imageB.jpg"))
        prespe_ransac_btn1 = QPushButton("Perspective RANSAC pair1")
        prespe_ransac_btn1.clicked.connect(lambda: self.run_perspective_runsac("Resources/pair1_imageA.jpg", "Resources/pair1_imageB.jpg"))
        prespe_ransac_btn2 = QPushButton("Perspective RANSAC pair2")
        prespe_ransac_btn2.clicked.connect(lambda: self.run_perspective_runsac("Resources/pair2_imageA.jpg", "Resources/pair2_imageB.jpg"))
        prespe_ransac_btn3 = QPushButton("Perspective RANSAC pair3")
        prespe_ransac_btn3.clicked.connect(lambda: self.run_perspective_runsac("Resources/pair3_imageA.jpg", "Resources/pair3_imageB.jpg"))
        buttonsVBox.addWidget(warping_btn)
        buttonsVBox.addWidget(affine_ransac_btn1)
        buttonsVBox.addWidget(affine_ransac_btn2)
        buttonsVBox.addWidget(affine_ransac_btn3)
        buttonsVBox.addWidget(prespe_ransac_btn1)
        buttonsVBox.addWidget(prespe_ransac_btn2)
        buttonsVBox.addWidget(prespe_ransac_btn3)
        buttonsWidget = QWidget()
        buttonsWidget.setLayout(buttonsVBox)
        mainHBox.addWidget(buttonsWidget)
        self.setLayout(mainHBox)

    def run_warping(self):
        res = run_warp()
        height, width, bytesPerComponent = res.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(res, cv2.COLOR_BGR2RGB, res)
        QImg = QImage(res.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.image_box.setPixmap(pixmap)

    def run_affine_runsac(self, img1, img2):
        res = run_ransac(img1, img2)
        self.set_images(res, img1, img2)

    def run_perspective_runsac(self, img1, img2):
        res = run_per_ransac(img1, img2)
        self.set_images(res, img1, img2)

    def set_images(self, res, img1, img2):
        self.orig_img1.setPixmap(QPixmap(img1))
        self.orig_img2.setPixmap(QPixmap(img2))
        height, width, bytesPerComponent = res.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(res, cv2.COLOR_BGR2RGB, res)
        QImg = QImage(res.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.image_box.setPixmap(pixmap)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    wnd = RANSAC()
    wnd.show()
    sys.exit(app.exec_())