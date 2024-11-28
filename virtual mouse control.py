import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 1.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1.5)
cTime = 0
pTime = 0
detector = htm.HandDetector(detectionCon=0.9)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))

    cv2.imshow("virtual mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()