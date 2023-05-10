import cv2
import numpy as np
from numpy.random import multinomial
from scipy.interpolate import UnivariateSpline

def create_LUT_8UC1( x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))
    
def cooling_transform(img):
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])
    
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

def warming_transform( img):
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])
    
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

def random_transform( img):

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

def mirror_transform( img):
    num_rows, num_cols = img.shape[:2]

    src_points = np.float32([[0,0], [num_cols-1,0], [0,num_rows-1]])
    dst_points = np.float32([[num_cols-1,0], [0,0], [num_cols-1,num_rows-1]])
    matrix = cv2.getAffineTransform(src_points, dst_points)
    img_afftran = cv2.warpAffine(img, matrix, (num_cols,num_rows))
    
    return img_afftran
    
def bw_transform( img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def vapourwave_transform( img):

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

def edge_detection_transform( img):

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


def special_transform( img):

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

    
def gaussian_blur_transform( img):
        
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

    
def pink_transform( img):

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

def sepia_transorm( img):

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