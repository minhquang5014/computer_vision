import cv2
import numpy as np
import utils
webcam = False
path = 'images_and_video/photo_0807_15h11m58s.jpg'

cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)

wP = 210
hP = 297

scale = 3

while True:
    if webcam: 
        success, img = cap.read()
        img = cv2.flip(img, 1)
    else: 
        img = cv2.imread(path)
    
    img, findContour = utils.getContours(img, showCanny=True, minArea=20000, draw = True)
    if len(findContour)!=0:
        biggest = findContour[0][2]
        # print(biggest)
        utils.warpImg(img, biggest, wP, hP)
        # cv2.imshow('A4', imgWarp)
    cv2.imshow('original', img)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()