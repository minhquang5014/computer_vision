import numpy as np
import dlib
import cv2
import threading

def empty(a):
    pass

cv2.namedWindow("Adjustments")
cv2.resizeWindow("Adjustments", 640, 240)
cv2.createTrackbar("Blue", 'Adjustments', 0, 255, empty)
cv2.createTrackbar("Green", 'Adjustments', 0, 255, empty)
cv2.createTrackbar("Red", 'Adjustments', 0, 255, empty)
cv2.createTrackbar("Brightness", 'Adjustments', 50, 100, empty)
cv2.createTrackbar("Contrast", 'Adjustments', 50, 100, empty)

def createBox(img, points, scale=1.0, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)
    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
        return imgCrop
    else:
        return mask

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    brightness = (brightness - 50) * 2
    contrast = (contrast - 50) * 2

    B = brightness / 100.0
    C = contrast / 100.0
    k = np.tan((45 + 44 * C) / 180 * np.pi)

    img = (image - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

class VideoCaptureThread(threading.Thread):
    def __init__(self, src=0):
        super(VideoCaptureThread, self).__init__()
        self.cap = cv2.VideoCapture(src)
        self.success, self.frame = self.cap.read()
        self.stopped = False

    def run(self):
        while not self.stopped:
            self.success, self.frame = self.cap.read()

    def read(self):
        return self.success, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def process_frame(frame, detector, predictor):
    img = cv2.flip(frame, 1)
    img = cv2.resize(img, (640, 480))
    imgOriginal = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    myPoints = []

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(imgGray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])
        myPoints = np.array(myPoints)

        imgLips = createBox(img, myPoints[48:61], 3, True, False)
        imgColorLips = np.zeros_like(imgLips)
        b = cv2.getTrackbarPos('Blue', 'Adjustments')
        g = cv2.getTrackbarPos('Green', 'Adjustments')
        r = cv2.getTrackbarPos('Red', 'Adjustments')
        imgColorLips[:] = b, g, r
        imgColorLips = cv2.bitwise_and(imgColorLips, imgLips)
        imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
        imgOriginalGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
        imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)
        imgColorLips = cv2.addWeighted(imgOriginalGray, 1, imgColorLips, 0.4, 0)
        
        brightness = cv2.getTrackbarPos('Brightness', 'Adjustments')
        contrast = cv2.getTrackbarPos('Contrast', 'Adjustments')
        imgAdjusted = adjust_brightness_contrast(imgOriginal, brightness, contrast)
        
        return imgColorLips, imgOriginal, imgAdjusted

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:/Users/Admin/Desktop/image/facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat")

cap_thread = VideoCaptureThread()
cap_thread.start()

while True:
    success, frame = cap_thread.read()
    if not success:
        break

    imgColorLips, imgOriginal, imgAdjusted = process_frame(frame, detector, predictor)
    
    cv2.imshow('Adjustments', imgColorLips)
    cv2.imshow("Original", imgOriginal)
    cv2.imshow("Adjusted", imgAdjusted)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap_thread.stop()
        break

cv2.destroyAllWindows()
