import mediapipe as mp
import cv2
import time
import os
import warnings

# Suppress TensorFlow and other library warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses TensorFlow info and warning messages

# Optionally suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

cap = cv2.VideoCapture(1)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w , c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id ==0:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), cv2.FILLED)
            mpDraw.draw_landmarks(frame, 
                                  handLms, 
                                  mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
    cv2.imshow("hand detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()