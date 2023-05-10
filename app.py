import sys
import os
import tempfile
import numpy as np
from filters import *
from numpy.random import multinomial
import cv2
from scipy.interpolate import UnivariateSpline
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

        self.path = ""
        self.filename = ""

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet('''
            QLabel{
                border: 2px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)

    def set_new_image(self, file_path):
        self.path = file_path
        self.filename = os.path.basename(file_path)
        pixmap = QPixmap(file_path)
        self.setPixmap(pixmap.scaled(
            400, 400, Qt.AspectRatioMode.KeepAspectRatio))
    
    

    def set_image(self, file_path):
        self.path = file_path
        self.filename = os.path.basename(file_path)
        pixmap = QPixmap(file_path)
        self.setPixmap(pixmap.scaled(
            400, 400, Qt.AspectRatioMode.KeepAspectRatio))
    
    def save_image(self,temporary_path):
        base = os.path.basename(temporary_path)
        homedir = os.path.expanduser('~')
        new_file_path = os.path.join(homedir, 'Pictures', base)
        # print(new_file_path)
        # 
        # base, ext = os.path.splitext(new_file_path)
        # count = 1
        # while os.path.exists(new_file_path):
        #     new_file_path = f"{base}_{count}{ext}"
        os.rename(temporary_path, new_file_path)

       

        #return new_file_path  





class DraggableImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.path = ""
        self.filename = ""
        self.setAcceptDrops(True)


        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.setText('\n\n Vea pilt siia \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 2px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)
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
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.set_image(file_path)
                event.accept()
                return
        
        event.ignore()

      
    def set_image(self, file_path):
        self.path = file_path
        self.filename = os.path.basename(file_path)
        pixmap = QPixmap(file_path)
        self.setPixmap(pixmap.scaled(
            400, 400, Qt.AspectRatioMode.KeepAspectRatio))
    
    def set_temporary_image(self,path, new_image, filter_name):

        ext = path.split('.')[-1]
        new_file_path = '.'.join(self.filename.split(
            '.')[:-1]) + ' (' + filter_name + ').' + ext
        tempdir = tempfile.gettempdir()
        new_file_path = os.path.join(tempdir, new_file_path)
        cv2.imwrite(new_file_path, new_image)

        #print('this is the new tmp path', new_file_path )

        return new_file_path



class KernelGram(QWidget):
    def __init__(self):
        super().__init__()


        self.resize(900, 500)
        self.setAcceptDrops(True)
        self.temporary_path = ''

        mainLayout: QVBoxLayout = QVBoxLayout()

        # Filter row
        label = QLabel()
        label.setText('Vali filter: ')
        label.adjustSize()

        self.choice = QComboBox()

        for f in ['', 'Seepia', 'Roosa', 'Soe', 'Külm', 'Udune', 'Vapourwave', 'Servad', 'Must-valge', 'Peegelpilt', 'Suvaline', 'Tekstituvastus']:
            self.choice.addItem(f)

        self.choice.currentIndexChanged.connect(self.on_click)
        # Photo row
        self.photoViewer = DraggableImageLabel()
        self.photoViewer2 = ImageLabel()

        self.photoViewer.setMinimumHeight(400)
        self.photoViewer2.setMinimumHeight(400)

        photo_row = QHBoxLayout()
        photo_row.addWidget(self.photoViewer)
        photo_row.addWidget(self.photoViewer2)

        save_button = QPushButton()
        save_button.setText('Salvesta pilt')
        save_button.clicked.connect(self.save_wrap)


       
        refresh_button = QPushButton()
        refresh_button.setText('Uuenda')
        refresh_button.clicked.connect(self.on_click)

        filter_row = QHBoxLayout()
        filter_row.addWidget(label)
        filter_row.addWidget(self.choice)
        filter_row.addStretch()
        filter_row.addWidget(refresh_button)
        filter_row.addWidget(save_button)

        mainLayout.addLayout(filter_row)


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
        if self.photoViewer.path == '':
            return

        # vastavalt comboboxi valikule kasutab erinevat filtrit
        selected_filter = self.choice.currentText()

        img = cv2.imread(self.photoViewer.path)

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
        elif selected_filter == 'Tekstituvastus':
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
        self.temporary_path = self.photoViewer.set_temporary_image(self.photoViewer.path, img, selected_filter)
        self.photoViewer2.set_image(self.temporary_path)

    def save_wrap(self):

        if os.path.exists(self.temporary_path):
            #print(self.temporary_path)
            self.photoViewer2.save_image(self.temporary_path)
        else:
            print("Error", "Temporary file does not exist")
    
    
    
   


app = QApplication(sys.argv)
gram = KernelGram()
gram.show()
sys.exit(app.exec())
