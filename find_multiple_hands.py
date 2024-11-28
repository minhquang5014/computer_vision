import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 1.4)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1.4)
cTime = 0
pTime = 0

detector = HandDetector(detectionCon=0.9, maxHands=2)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    hands, frame = detector.findHands(frame, flipType=False)
    if hands:
        # get first hand
        hand1 = hands[0]
        lmList1 = hand1["lmList"] # list of 21 landmark points
        bbox1 = hand1["bbox"]   # bounding box info
        centerPoint1 = hand1["center"]   # get the center of the hand cx and cy
        handType1 = hand1["type"]   # whether it's left or right
        fingers1 = detector.fingersUp(hand1)
       
        if len(lmList1) >= 13:  # Ensure landmarks exist
            result = detector.findDistance(lmList1[8], lmList1[12], frame)
            
            # Check if result is not None and handle multiple return values
            if result:
                if len(result) == 2:
                    l, info = result  # Unpack if it returns (l, info)
                    print(l, info)
                elif len(result) == 3:
                    l, info, frame = result  # Unpack if it returns (l, info, frame)
                    print(l, info)

        if len(hands) == 2:
            hand2 = hands[0]
            lmList2 = hand2["lmList"] # list of 21 landmark points
            bbox2 = hand2["bbox"]   # bounding box info
            centerPoint2 = hand2["center"]   # get the center of the hand cx and cy
            handType2 = hand2["type"]   # whether it's left or right
            fingers2 = detector.fingersUp(hand2)  
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
    cv2.imshow("distance hand measurement", frame)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()