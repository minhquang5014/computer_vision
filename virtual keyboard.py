import cv2
import numpy as np
import time
from HandTrackingModule import HandDetector  # Ensure this matches the actual module and class names
from time import sleep
import cvzone

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 1.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1.5)
cTime = 0
pTime = 0
detector = HandDetector(detectionCon=0.9)

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
        ["Del"],
        ["En"],
        ["Sp"]]
   
finalText = ""
def draw_all(frame, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(frame, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(frame, button.text, (x + 10, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    return frame

class Button:
    def __init__(self, pos, text, size=[60, 80]):
        self.pos = pos
        self.text = text
        self.size = size

buttonList = []
for i, row in enumerate(keys):
    for j, key in enumerate(row):
        if key == "Del":
            size = [100, 80]
        elif key =="En":
            size = [100, 80]
        elif key == "Sp":
            size = [100, 80]
        else:
            size = [60, 80]
        buttonList.append(Button([20 + 75 * j, 100 + 100 * i], key))


while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    frame = detector.findHands(frame)  # Detect hands
    lmList = detector.findPosition(frame, draw=False)  # Get landmark positions
    frame = draw_all(frame, buttonList)
    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            # Check if the index finger tip is within the button area
            if x < lmList[8][1] < x + w and y < lmList[8][2] < y + h:
                cv2.rectangle(frame, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                cv2.putText(frame, button.text, (x + 10, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                # Check the distance between the index finger tip and middle finger tip
                l, _, _ = detector.findDistance(8, 12, lmList, frame)
                if l<40:
                    cv2.rectangle(frame, button.pos, (x+w, y+h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, button.text, (x+10, y+58), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    if button.text == "Del":
                        finalText = finalText[:-1]
                    elif button.text == "En":
                        finalText = ""
                    elif button.text == "Sp":
                        finalText += " "
                    else:
                        finalText += button.text
                    sleep(0.4)
    
    cv2.rectangle(frame, (160, 400), (900, 480), (175, 0, 175), cv2.FILLED)
    cv2.putText(frame, finalText, (170, 460), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
    cv2.imshow("Virtual Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
