import cv2
import numpy as np
from numpy.random import multinomial
from scipy.interpolate import UnivariateSpline

def create_LUT_8UC1(x, y):
    # Create a lookup table (LUT) using spline interpolation
    spl = UnivariateSpline(x, y)
    return spl(range(256))

def apply_filter(img, kernel):
    # Apply a filter kernel to the image
    img_filtered = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    
    # Normalize pixel values
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return img_filtered

def cooling_transform(img):
    # Create lookup tables for channel transformations
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])

    # Apply channel transformations for blue and red channels
    c_b, c_g, c_r = cv2.split(img)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    img_bgr_cold = cv2.merge((c_b, c_g, c_r))

    # Decrease color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_cold, cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
    img_bgr_cold = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)

    return img_bgr_cold

def warming_transform(img):
    # Create lookup tables for channel transformations
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])

    # Split the image channels
    c_b, c_g, c_r = cv2.split(img)

    # Apply channel transformations
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)

    # Merge the channels back into an image
    img_bgr_warm = cv2.merge((c_b, c_g, c_r))

    # Apply color saturation increase
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_warm, cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)

    # Merge the channels back into an image
    img_bgr_warm = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)

    return img_bgr_warm

def random_transform(img):

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

def mirror_transform(img):
    # Mirror the image horizontally
    img_mirror = cv2.flip(img, 1)

    return img_mirror

def bw_transform(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_gray

def vapourwave_transform(img):
    # Define the kernel for the filter
    kernel = np.matrix([
        [-1, 1, -1],
        [1, 0, 1],
        [-1, 1, -1]
    ])

    # Convert the image data to floating-point values for precise calculations
    img = np.array(img, dtype=np.float64)

    # Apply the filter using cv2.transform
    img = cv2.transform(img, kernel)

    # Normalize the pixel values and convert back to uint8 data type
    img[np.where(img > 255)] = 255
    img = np.array(img, dtype=np.uint8)

    return img



def edge_detection_transform(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(image=img_gray, threshold1=80, threshold2=200)

    # Convert the edges back to BGR color format
    img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return img_edges

def special_transform(img):
    # Create a filter kernel for special effect
    kernel = np.matrix([
        [-1, 1, -1],
        [1, 0, 1],
        [-1, 1, -1]
    ])

    # Apply the filter kernel to the image
    img = apply_filter(img, kernel)

    return img

def gaussian_blur_transform(img):
    t = np.linspace(-10, 10, 30)
    bump = np.exp(-0.1*t**2)
    bump /= np.trapz(bump)

    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

    # Convert the image to float64 for precise calculations
    img = np.array(img, dtype=np.float64)

    # Apply the filter
    img = cv2.filter2D(img, ddepth=-1, kernel=kernel)

    # Normalize the values and convert back to uint8
    img[np.where(img > 255)] = 255
    img = np.array(img, dtype=np.uint8)

    return img



def pink_transform( img):

    kernel = np.matrix([
        [0.7, 0.5, 0.7],
        [0.5, 0.3, 0.5],
        [0.7, 0.5, 0.7]
    ])

    # For percision
    img = np.array(img, dtype=np.float64)

    # apply filter
    img = cv2.transform(img, kernel)

    # Normalize
    img[np.where(img > 255)] = 255
    img = np.array(img, dtype=np.uint8)

    return img




def sepia_transform(img):
    sepia_kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])

    # Convert the image to floating point for precise transformations
    img = img.astype(np.float32) / 255.0

    # Apply the sepia filter by matrix multiplication with the kernel
    img = np.matmul(img, sepia_kernel.T)

    # Clip the values to the valid range [0, 1]
    img = np.clip(img, 0, 1)

    # Convert the image back to the original data type (uint8)
    img = (img * 255).astype(np.uint8)

    return img

def detect_faces(img):
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img






def clarendon_filter(img):
    # Increase contrast and apply a blue tint
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0

    img[:, :, 0] = np.minimum(1, img[:, :, 0] * 1.2)
    img[:, :, 1] = np.minimum(1, img[:, :, 1] * 1.1)
    img[:, :, 2] = np.minimum(1, img[:, :, 2] * 0.9)

    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def gingham_filter(img):
    # Apply a vintage, faded look with a slight green tint
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0

    img[:, :, 0] = np.minimum(1, img[:, :, 0] * 0.9)
    img[:, :, 1] = np.minimum(1, img[:, :, 1] * 1.1)
    img[:, :, 2] = np.minimum(1, img[:, :, 2] * 0.9)

    img[:, :, [0, 2]] += 0.02

    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def juno_filter(img):
    # Apply a warm, vintage look with increased contrast
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0

    img[:, :, 0] = np.minimum(1, img[:, :, 0] * 1.2)
    img[:, :, 1] = np.minimum(1, img[:, :, 1] * 1.1)
    img[:, :, 2] = np.minimum(1, img[:, :, 2] * 0.9)

    img[:, :, [0, 2]] += 0.03

    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def lark_filter(img):
    # Apply a cool, desaturated look with increased brightness
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0

    img[:, :, 0] = np.minimum(1, img[:, :, 0] * 0.9)
    img[:, :, 1] = np.minimum(1, img[:, :, 1] * 1.1)
    img[:, :, 2] = np.minimum(1, img[:, :, 2] * 1.1)

    img[:, :, 1] -= 0.05
    img[:, :, 2] -= 0.1

    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def mayfair_filter(img):
    # Apply a warm, vintage look with increased contrast and saturation
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0

    img[:, :, 0] = np.minimum(1, img[:, :, 0] * 1.1)
    img[:, :, 1] = np.minimum(1, img[:, :, 1] * 1.15)
    img[:, :, 2] = np.minimum(1, img[:, :, 2] * 0.95)

    img[:, :, 0] += 0.05
    img[:, :, 1] += 0.05

    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def sierra_filter(img):
    # Apply a faded, nostalgic look with a cool tone
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0

    img[:, :, 0] = np.minimum(1, img[:, :, 0] * 1.1)
    img[:, :, 1] = np.minimum(1, img[:, :, 1] * 1.1)
    img[:, :, 2] = np.minimum(1, img[:, :, 2] * 0.9)

    img[:, :, 0] -= 0.05
    img[:, :, 2] += 0.05

    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def valencia_filter(img):
    # Apply a warm, romantic look with increased contrast and saturation
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0

    img[:, :, 0] = np.minimum(1, img[:, :, 0] * 1.1)
    img[:, :, 1] = np.minimum(1, img[:, :, 1] * 1.15)
    img[:, :, 2] = np.minimum(1, img[:, :, 2] * 0.95)

    img[:, :, 0] += 0.03
    img[:, :, 1] += 0.02

    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def object_detection(img):  
    image = img

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open('./models/yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet('./models/yolov3.weights', './models/yolov3.cfg')

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)

    return img







