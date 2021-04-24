from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from exeA import *
from exeB import *
from exeBiDirectionalTest import *
from PyQt5.QtCore import Qt


class SIFT(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIFT")
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
        warping_btn = QPushButton("Show University Key Points")
        warping_btn.clicked.connect(self.university)
        ratio_sift_btn1 = QPushButton("Key Points matching by RatioTest pair1")
        ratio_sift_btn1.clicked.connect(lambda: self.match_points_ratio("Resources/pair1_imageA.jpg", "Resources/pair1_imageB.jpg"))
        ratio_sift_btn2 = QPushButton("Key Points matching by RatioTest pair2")
        ratio_sift_btn2.clicked.connect(lambda: self.match_points_ratio("Resources/pair2_imageA.jpg", "Resources/pair2_imageB.jpg"))
        ratio_sift_btn3 = QPushButton("Key Points matching by RatioTest pair3")
        ratio_sift_btn3.clicked.connect(lambda: self.match_points_ratio("Resources/pair3_imageA.jpg", "Resources/pair3_imageB.jpg"))
        bidir_sift_btn1 = QPushButton("Key Points matching by Bi-Directional pair1")
        bidir_sift_btn1.clicked.connect(lambda: self.match_points_bi("Resources/pair1_imageA.jpg", "Resources/pair1_imageB.jpg"))
        bidir_sift_btn2 = QPushButton("Key Points matching by Bi-Directional pair2")
        bidir_sift_btn2.clicked.connect(lambda: self.match_points_bi("Resources/pair2_imageA.jpg", "Resources/pair2_imageB.jpg"))
        bidir_sift_btn3 = QPushButton("Key Points matching by Bi-Directional pair3")
        bidir_sift_btn3.clicked.connect(lambda: self.match_points_bi("Resources/pair3_imageA.jpg", "Resources/pair3_imageB.jpg"))
        buttonsVBox.addWidget(warping_btn)
        buttonsVBox.addWidget(ratio_sift_btn1)
        buttonsVBox.addWidget(ratio_sift_btn2)
        buttonsVBox.addWidget(ratio_sift_btn3)
        buttonsVBox.addWidget(bidir_sift_btn1)
        buttonsVBox.addWidget(bidir_sift_btn2)
        buttonsVBox.addWidget(bidir_sift_btn3)
        buttonsWidget = QWidget()
        buttonsWidget.setLayout(buttonsVBox)
        mainHBox.addWidget(buttonsWidget)
        self.setLayout(mainHBox)

    def university(self):
        res = university_keypoints()
        self.set_images(res)

    def match_points_ratio(self, img1, img2):
        res = keyoints_matching_ratio(img1, img2)
        self.set_images(res, img1, img2)

    def match_points_bi(self, img1, img2):
        res = keyoints_matching_bi(img1, img2)
        self.set_images(res, img1, img2)

    def set_images(self, res, img1=None, img2=None):
        if img1 and img2:
            self.orig_img1.setPixmap(QPixmap(img1))
            self.orig_img2.setPixmap(QPixmap(img2))
        height, width, bytesPerComponent = res.shape
        bytesPerLine = 3 * width
        QImg = QImage(res.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.image_box.setPixmap(pixmap)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    wnd = SIFT()
    wnd.show()
    sys.exit(app.exec_())