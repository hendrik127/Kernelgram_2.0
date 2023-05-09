import sys
import os
import tempfile
from filters import *
import cv2
import pytesseract
# from TextExtractApi.TextExtract import TextExtractFunctions
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QHBoxLayout, QComboBox, QPlainTextEdit
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText('\n\n Vea pilt siia \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 2px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()

        self.path = ""
        self.path2 = ""

        self.filename = ""
        self.filename2 = ""

        self.resize(900, 500)
        self.setAcceptDrops(True)

        mainLayout: QVBoxLayout = QVBoxLayout()

        # Filter row
        label = QLabel()
        label.setText('Vali filter: ')
        label.adjustSize()

        self.choice = QComboBox()

        for f in ['', 'Seepia', 'Roosa', 'Soe', 'Külm', 'Udune', 'Vapourwave', 'Servad', 'Must-valge', 'Peegelpilt', 'Suvaline', 'Teksti tuvastus']:
            self.choice.addItem(f)

        self.choice.currentIndexChanged.connect(self.on_click)

        save_button = QPushButton()
        save_button.setText('Salvesta pilt')
        save_button.clicked.connect(self.save_permanent_new_image_by_path)

        refresh_button = QPushButton()
        refresh_button.setText('Uuenda')
        refresh_button.clicked.connect(self.refresh_new_image)

        filter_row = QHBoxLayout()
        filter_row.addWidget(label)
        filter_row.addWidget(self.choice)
        filter_row.addStretch()
        filter_row.addWidget(refresh_button)
        filter_row.addWidget(save_button)

        mainLayout.addLayout(filter_row)

        # Photo row
        self.photoViewer = ImageLabel()
        self.photoViewer2 = ImageLabel()

        self.photoViewer.setMinimumHeight(400)
        self.photoViewer2.setMinimumHeight(400)

        photo_row = QHBoxLayout()
        photo_row.addWidget(self.photoViewer)
        photo_row.addWidget(self.photoViewer2)

        # Text row

        self.textbox = QPlainTextEdit()
        # self.textbox.setMaximumSizeHint(100)

        text_row = QHBoxLayout()

        text_row.addWidget(self.textbox)

        mainLayout.addLayout(photo_row)
        mainLayout.addStretch()
        mainLayout.addLayout(text_row)

        self.setLayout(mainLayout)

        self.setWindowTitle("KernelGram")

    @pyqtSlot()
    def on_click(self):
        if self.path == '':
            return

        # vastavalt comboboxi valikule kasutab erinevat filtrit
        selected_filter = self.choice.currentText()

        img = cv2.imread(self.path)

        if selected_filter == 'Seepia':
            img = sepia_transorm(img)
        elif selected_filter == 'Roosa':
            img = pink_transform(img)
        elif selected_filter == 'Udune':
            img = gaussian_blur_transform(img)
        elif selected_filter == 'Special':
            img = special_transform(img)
        elif selected_filter == 'Vapourwave':
            img = vapourwave_transform(img)
        elif selected_filter == 'Servad':
            img = edge_detection_transform(img)
        elif selected_filter == 'Teksti tuvastus':
            self.textbox.setPlainText(pytesseract.image_to_string(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        elif selected_filter == 'Must-valge':
            img = bw_transform(img)
        elif selected_filter == 'Peegelpilt':
            img = mirror_transform(img)
        elif selected_filter == 'Suvaline':
            img = random_transform(img)
        elif selected_filter == 'Soe':
            img = warming_transform(img)
        elif selected_filter == 'Külm':
            img = cooling_transform(img)
        else:
            return

        # salvestame pildi ja näitame seda aknas
        self.display_new_image(img, selected_filter)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.DropAction.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)

            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.path = file_path
        self.filename = os.path.basename(file_path)

        pixmap = QPixmap(file_path)
        aspect_ratio = pixmap.width() / pixmap.height()
        image_width = self.photoViewer.width() - 8

        self.photoViewer.setPixmap(pixmap.scaled(
            400, 400, Qt.AspectRatioMode.KeepAspectRatio))

    def set_new_image(self, file_path):
        self.path2 = file_path
        self.filename2 = os.path.basename(file_path)

        pixmap = QPixmap(file_path)
        aspect_ratio = pixmap.width() / pixmap.height()
        image_width = self.photoViewer2.width() - 8

        self.photoViewer2.setPixmap(pixmap.scaled(
            400, 400, Qt.AspectRatioMode.KeepAspectRatio))

    def display_new_image(self, new_image, filter_name):

        new_file_path = self.save_new_image(
            new_image, filter_name, temporary=True)

        # näitame uut pilti kasutajale
        self.set_new_image(new_file_path)

    def save_permanent_new_image_by_path(self):
        new_image_path = self.path2
        filter_name = self.choice.currentText()

        try:
            new_img = cv2.imread(new_image_path)

            return self.save_new_image(new_img, filter_name, False)
        except:
            print("There is no image")

            Exception('There is no new image')

            # return False

    def save_new_image(self, new_image, filter_name, temporary):

        ext = self.path.split('.')[-1]

        new_file_path = '.'.join(self.filename.split(
            '.')[:-1]) + ' (' + filter_name + ').' + ext

        if temporary:
            # salvesta ajutisse kausta
            tempdir = tempfile.gettempdir()

            new_file_path = os.path.join(tempdir, new_file_path)

            cv2.imwrite(new_file_path, new_image)
        else:
            homedir = os.path.expanduser('~')

            new_file_path = os.path.join(homedir, 'Pictures', new_file_path)

            # salvesta kausta
            cv2.imwrite(new_file_path, new_image)

        return new_file_path

    def refresh_new_image(self):
        self.on_click()


app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec())
