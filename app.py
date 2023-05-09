import sys
import os
import tempfile
import numpy as np
from numpy.random import multinomial
import cv2
from scipy.interpolate import UnivariateSpline
import pytesseract
#from TextExtractApi.TextExtract import TextExtractFunctions  
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

        for f in ['', 'Seepia', 'Roosa', 'Soe', 'Külm', 'Udune', 'Vapourwave', 'Servad', 'Must-valge', 'Peegelpilt', 'Suvaline','Teksti tuvastus']:
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
        #self.textbox.setMaximumSizeHint(100)

        text_row = QHBoxLayout()

        text_row.addWidget(self.textbox)


        mainLayout.addLayout(photo_row)
        mainLayout.addStretch()
        mainLayout.addLayout(text_row)

        self.setLayout(mainLayout)

        self.setWindowTitle("KernelGramm")

    @pyqtSlot()
    def on_click(self):
        if self.path == '':
            return

        # vastavalt comboboxi valikule kasutab erinevat filtrit
        selected_filter = self.choice.currentText()

        img = cv2.imread(self.path)

        if selected_filter == 'Seepia':
            img = self.sepia_transorm(img)
        elif selected_filter == 'Roosa':
            img = self.pink_transform(img)
        elif selected_filter == 'Udune':
            img = self.gaussian_blur_transform(img)
        elif selected_filter == 'Special':
            img = self.special_transform(img)
        elif selected_filter == 'Vapourwave':
            img = self.vapourwave_transform(img)
        elif selected_filter == 'Servad':
            img = self.edge_detection_transform(img)
        elif selected_filter == 'Teksti tuvastus':
            self.textbox.setPlainText(pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        elif selected_filter == 'Must-valge':
            img = self.bw_transform(img)
        elif selected_filter == 'Peegelpilt':
            img = self.mirror_transform(img)
        elif selected_filter == 'Suvaline':
            img = self.random_transform(img)
        elif selected_filter == 'Soe':
            img = self.warming_transform(img)
        elif selected_filter == 'Külm':
            img = self.cooling_transform(img)
        else:
            return
        
        # salvestame pildi ja näitame seda aknas
        self.display_new_image(img, selected_filter)
    
    def create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))
    
    def cooling_transform(self,img):
        incr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
        decr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])
        
        img_bgr_in = img
        c_b, c_g, c_r = cv2.split(img_bgr_in)
        c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
        img_bgr_cold = cv2.merge((c_b, c_g, c_r))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_cold,
            cv2.COLOR_BGR2HSV))
        c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
        img_bgr_cold = cv2.cvtColor(cv2.merge(
            (c_h, c_s, c_v)),
            cv2.COLOR_HSV2BGR)

        return img_bgr_cold

    def warming_transform(self, img):
        incr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
        decr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])
        
        img_bgr_in = img

        c_b, c_g, c_r = cv2.split(img_bgr_in)
        c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
        img_bgr_warm = cv2.merge((c_b, c_g, c_r))

        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)

        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_warm, cv2.COLOR_BGR2HSV))
        c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)

        img_bgr_warm = cv2.cvtColor(cv2.merge(
                (c_h, c_s, c_v)),
                cv2.COLOR_HSV2BGR)

        return img_bgr_warm

    def random_transform(self, img):

        # genereerib kaheksa suvalist arvu, millest pooled on positiivsed ja pooled negatiivsed.
        # ühe summa on 1, teisel -1 ehk kokku annavad 0
        # lisame need ühte arraysse, segame selle suvaliselt ära ja siis paigutame kernelisse

        random_pos = np.array(multinomial(100, [1/4.] * 4)/100)
        random_neg = np.array(multinomial(100, [1/4.] * 4)/(-100))
        random = np.append(random_pos, random_neg)
        np.random.shuffle(random)

        kernel = np.matrix([
            random[0:3],
            [random[3], 0, random[4]],
            random[5:]
        ])

        # muudab pildi andmed ujuvkoma arvudeks, et teisendused täpsed oleksid
        img = np.array(img, dtype=np.float64)

        # rakendame filtrit
        img = cv2.transform(img, kernel)

        # normaliseerime väärtused ja teisendame täisarvudeks tagasi
        img[np.where(img > 255)] = 255
        img = np.array(img, dtype=np.uint8)

        return img

    def mirror_transform(self, img):
        num_rows, num_cols = img.shape[:2]

        src_points = np.float32([[0,0], [num_cols-1,0], [0,num_rows-1]])
        dst_points = np.float32([[num_cols-1,0], [0,0], [num_cols-1,num_rows-1]])
        matrix = cv2.getAffineTransform(src_points, dst_points)
        img_afftran = cv2.warpAffine(img, matrix, (num_cols,num_rows))
        
        return img_afftran
    
    def bw_transform(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def vapourwave_transform(self, img):

        kernel = np.matrix([
            [-1, 1, -1],
            [1, 0, 1],
            [-1, 1, -1]
        ])

        # muudab pildi andmed ujuvkoma arvudeks, et teisendused täpsed oleksid
        img = np.array(img, dtype=np.float64)

        # rakendame filtrit
        img = cv2.transform(img, kernel)

        # normaliseerime väärtused ja teisendame täisarvudeks tagasi
        img[np.where(img > 255)] = 255
        img = np.array(img, dtype=np.uint8)

        return img

    def edge_detection_transform(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.matrix([
            [-2, -2, -2, -2, -2],
            [-2, -1, -1, -1, -2],
            [-2, -1, 40, -1, -2],
            [-2, -1, -1, -1, -2],
            [-2, -2, -2, -2, -2]
        ])

        # muudab pildi andmed ujuvkoma arvudeks, et teisendused täpsed oleksid
        # img = np.array(img, dtype=np.float64)

        # rakendame filtrit
        # img = cv2.filter2D(img, ddepth=-1, kernel=kernel)
        # img_blur = cv2.GaussianBlur(img, (3,3), 0)
        edges = cv2.Canny(image=img, threshold1=80, threshold2=200) # Canny Edge Detection
        
        img  = cv2.cvtColor(edges, cv2.IMREAD_COLOR)

        # normaliseerime väärtused ja teisendame täisarvudeks tagasi
        img[np.where(img > 255)] = 255
        img = np.array(img, dtype=np.uint8)

        return img


    def special_transform(self, img):

        kernel = np.matrix([
            [-1, 1, -1],
            [1, 0, 1],
            [-1, 1, -1]
        ])

        # muudab pildi andmed ujuvkoma arvudeks, et teisendused täpsed oleksid
        img = np.array(img, dtype=np.float64)

        # rakendame filtrit
        img = cv2.filter2D(img, ddepth=-1, kernel=kernel)

        # normaliseerime väärtused ja teisendame täisarvudeks tagasi
        img[np.where(img > 255)] = 255
        img = np.array(img, dtype=np.uint8)

        return img

    
    def gaussian_blur_transform(self, img):
        
        t = np.linspace(-10, 10, 30)
        bump = np.exp(-0.1*t**2)
        bump /= np.trapz(bump)

        kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
        # kernel = np.matrix(kernel)

        '''
        # cursed
        kernel = np.matrix([
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]
        ])


        # vapourwave
        
        '''

        # muudab pildi andmed ujuvkoma arvudeks, et teisendused täpsed oleksid
        img = np.array(img, dtype=np.float64)

        # rakendame filtrit
        img = cv2.filter2D(img, ddepth=-1, kernel=kernel)

        # normaliseerime väärtused ja teisendame täisarvudeks tagasi
        img[np.where(img > 255)] = 255
        img = np.array(img, dtype=np.uint8)

        return img

    
    def pink_transform(self, img):

        kernel = np.matrix([
            [0.7, 0.5, 0.7],
            [0.5, 0.3, 0.5],
            [0.7, 0.5, 0.7]
        ])

        # muudab pildi andmed ujuvkoma arvudeks, et teisendused täpsed oleksid
        img = np.array(img, dtype=np.float64)

        # rakendame filtrit
        img = cv2.transform(img, kernel)

        # normaliseerime väärtused ja teisendame täisarvudeks tagasi
        img[np.where(img > 255)] = 255
        img = np.array(img, dtype=np.uint8)

        return img

    def sepia_transorm(self, img):
        img = cv2.imread(self.path)

        seepia_kernel = np.matrix([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])

        # muudab pildi andmed ujuvkoma arvudeks, et teisendused täpsed oleksid
        img = np.array(img, dtype=np.float64)

        # rakendame sepia filtrit
        img = cv2.transform(img, seepia_kernel)

        # normaliseerime väärtused ja teisendame täisarvudeks tagasi
        img[np.where(img > 255)] = 255
        img = np.array(img, dtype=np.uint8)

        return img

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

        self.photoViewer.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
    
    def set_new_image(self, file_path):
        self.path2 = file_path
        self.filename2 = os.path.basename(file_path)

        pixmap = QPixmap(file_path)
        aspect_ratio = pixmap.width() / pixmap.height()
        image_width = self.photoViewer2.width() - 8

        self.photoViewer2.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
    
    def display_new_image(self, new_image, filter_name):

        new_file_path = self.save_new_image(new_image, filter_name, temporary=True)

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
        
        new_file_path = '.'.join(self.filename.split('.')[:-1]) + ' (' + filter_name + ').' + ext

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
