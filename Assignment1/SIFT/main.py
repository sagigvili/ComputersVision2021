from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from exeA import *
from exeB import *


class SIFT(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RANSAC")
        mainHBox = QHBoxLayout()
        self.image_box = QLabel()
        mainHBox.addWidget(self.image_box)
        buttonsVBox = QVBoxLayout()
        warping_btn = QPushButton("Show University Key Points")
        warping_btn.clicked.connect(self.university)
        affine_ransac_btn = QPushButton("Key Points matching")
        affine_ransac_btn.clicked.connect(self.match_points)
        buttonsVBox.addWidget(warping_btn)
        buttonsVBox.addWidget(affine_ransac_btn)
        buttonsWidget = QWidget()
        buttonsWidget.setLayout(buttonsVBox)
        mainHBox.addWidget(buttonsWidget)
        self.setLayout(mainHBox)

    def university(self):
        res = university_keypoints()
        self.set_images(res)

    def match_points(self):
        res = keyoints_matching()
        self.set_images(res)

    def set_images(self, res):
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