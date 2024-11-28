import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
wcam, hcam = 640, 480
cap.set(3, wcam)
cap.set(4, hcam)
cTime = 0
pTime = 0
detector = htm.HandDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
print(volume.GetVolumeRange())
volume.SetMasterVolumeLevel(volRange[0], None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 240
volPer = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList)!=0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2     # coordinates of the central point
        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)  # draw the circle around the central point
        distance = math.hypot(x2-x1, y2-y1)   # calculate the distance between the finger points
        print(distance)
        # hand range 35, 240
        # volume range -65.25 to 0.0
        vol = np.interp(distance, [35, 240], [minVol, maxVol])
        volBar = np.interp(distance, [35, 240], [400, 150])
        volPer = np.interp(distance, [35, 240], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        if distance < 35:
            cv2.circle(frame, (cx, cy), 10, (178, 74, 50), cv2.FILLED)
        if distance > 240:
            cv2.circle(frame, (cx, cy), 10, (0, 194, 252), cv2.FILLED)
    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(frame, (50, int(volBar)), (85, 400), (255, 255, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
    cv2.putText(frame, f'{int(volPer)}', (20 , 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
    cv2.imshow("Control Volumn", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()