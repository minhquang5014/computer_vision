import mediapipe as mp
import cv2
import warnings
import time
import math
# Suppress specific warnings from google.protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComp = modelComp

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, frame, draw=True):
        self.imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
                if draw:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 255), cv2.FILLED)
        return lmList
    def findDistance(self, p1, p2, lmList, frame):
        x1, y1 = lmList[p1][1], lmList[p1][2]
        x2, y2 = lmList[p2][1], lmList[p2][2]
        cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        distance = math.hypot(x2 - x1, y2 - y1)
        return distance, p1, p2
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    wcam, hcam = 640, 480
    cap.set(3, wcam)
    cap.set(4, hcam)

    detector = HandDetector(detectionCon=0.7)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("the camera might have been used for other purposes")
            break
        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False)
        if len(lmList) != 0:
            print(lmList[4], lmList[8])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime  # Update pTime after computing fps

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
        cv2.imshow("Control Volume", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
