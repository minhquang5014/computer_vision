import cv2
import numpy as np
from HandTrackingModule import HandDetector
import time
import random
import cvzone

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 1.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1.5)
detector = HandDetector(detectionCon=0.9)
colorR = (255, 0, 255)
cTime = 0
pTime = 0
cx, cy, w, h = 200, 200, 150, 150


class Dragrect:
    def __init__(self, posCenter, size=[150, 150]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # if the index finger tip is in the rectangle area
        if cx - w // 2 < cursor[1] < cx + w // 2 and cy - h // 2 < cursor[2] < cy + h // 2:
            self.posCenter = cursor[1], cursor[2]

rectList = []
for x in range(5):
    rectList.append(Dragrect([x * 150 + 160, 150]))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if lmList:
        l, _, _ = detector.findDistance(8, 12, lmList, frame)
        if l < 40:
            cursor = lmList[8]
            # call the update in here
            for rect in rectList:
                rect.update(cursor)
    
    #this is draw for transperancy
    imgNew = np.zeros_like(frame, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
    out=frame.copy()
    alpha=0.1
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, imgNew, 1 - alpha, 0)[mask]
  
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(out, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
    cv2.imshow("Virtual drag and drop", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
