import cv2
import numpy as np
import requests
import time
import face_recognition
# manage a queue, which is a data structure that follows the first-in-first-out principle
import queue
import os
# threads are the way to achieve concurrency in the programs, 
# allowing multiple operations to run simultaneously within the same process
import threading

import cvlib as cv
from matplotlib import pyplot as plt
from cvlib.object_detection import draw_bbox

path = 'images_and_video'
images = []
classNames = []
myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    if cls.endswith('avi') or cls.endswith('mp4'):
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

def findEncodeing(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
        return encodeList
encodeListKnown = findEncodeing(images)

class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)

        # initialization - create a FIFO queue
        # maxsize parameter, if maxsize is less than or equal 0, the queue size is infinite
        # store video frames
        self.q = queue.Queue() 

        # create threading for self._reader function 
        # run simultaneously with the video capture to read frames
        t = threading.Thread(target=self._reader)
        t.daemon = True

        # start threading
        t.start() 

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("cannot retrieve the video streaming")
                break
            if not self.q.empty:
                try:
                    self.q.get_nowait() # the oldest frame is removed to make room for the new one
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
    
URL = 'http://192.168.1.121'
cap = cv2.VideoCapture(URL + ":81/stream")

if __name__ == '__main__':
    requests.get(URL + "/control?var=framesize&val={}".format(8))

    while True:
        if cap.isOpened():
            ret, output_image = cap.read()
            start = time.time()
            # cv2.imshow("output", frame)
            # output_image = cv2.rotate(frame, cv2.ROTATE_180)
            # imgnp = np.array(bytearray(cap.read()), dtype=np.uint8)
            # img = cv2.imdecode(imgnp, -1)
            # bbox, label, conf = cv.detect_common_objects(frame)
            # output_image = draw_bbox(frame, bbox, label, conf, colors=(0, 255, 255))
            imgS = cv2.resize(output_image, (0, 0),None, 1/2, 1/2)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS)
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(output_image, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(output_image, name, (x1+6, y2-6), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            end = time.time()
            total_time = end - start
            if total_time != 0:
                fps = 1 / total_time
            cv2.putText(output_image, f'FPS: {int(fps)}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow("img", output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()