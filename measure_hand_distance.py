import cv2
import numpy as np
import HandTrackingModule as htm
import time
import cvzone
import mediapipe
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 1.4)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1.4)
cTime = 0
pTime = 0

detector = htm.HandDetector(detectionCon=0.9)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    hands, frame = detector.findHands(frame, draw=False)
    if hands:
        hands[0]
    
    cv2.imshow("distance hand measurement", frame)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
