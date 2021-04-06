from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, \
    QMessageBox
from PyQt5.QtGui import QPixmap


class Hough(QWidget):
    def __init__(self):
        super().__init__()
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
            pass
        else:
            QMessageBox.critical(self, "Error", "Please upload image first")

    def upload_image(self):
        image_path = QFileDialog.getOpenFileName(self, caption="File to Import",
                                           directory=".", filter="Jpeg Image (*.jpg)")
        if not image_path[0] == "":
            pixmap = QPixmap(image_path[0])
            self.image_box.setPixmap(pixmap)



if __name__== '__main__':
    import sys
    app = QApplication(sys.argv)
    wnd = Hough()
    wnd.show()
    sys.exit(app.exec_())