import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'images_and_video'
images = []   # list to store loaded images
classNames = []    # list to store names of the images
myList = os.listdir(path)   # access the image path directories

for cls in myList:   # loop through every image in the image directory
    curImg = cv2.imread(f'{path}/{cls}')   # read images
    if cls.endswith('.avi') or cls.endswith('.mp4'):
        continue
    images.append(curImg) # add images to the list
    classNames.append(os.path.splitext(cls)[0]) # add name of the filename (without extension) to the classNames list

def findEncoding(images):
    encodeList = []
    for img in images:   # loop through every image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # convert images from bgr to rgb
        encodes = face_recognition.face_encodings(img)
        if encodes:   
            encodeList.append(encodes[0])   # add the first face encoding to the list
    return encodeList   
encodeListKnown = findEncoding(images)   # get the face enconding for all images
print(encodeListKnown)
def markAttendance(name):
    """This function is for reading the student names and writing it onto the csv file is not detected""" 
    with open('Attendance.csv', 'r+') as f:   # open the attendance file in read and write mode
        myDataList = f.readlines()   # read all lines from the file
        nameList = []   # list to store name
        for line in myDataList:   # for every line in the csv file
            entry = line.split(',')    # split the column by comma
            nameList.append(entry[0])   # add to the name list the first element of entry
        if name not in nameList:   # if name is not on the name list
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')    # write name and time in the new line

cap = cv2.VideoCapture(0)
while True:   
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgS = cv2.resize(img, (0, 0), None, 1/2, 1/2)  # resize the images by a quarter of its orginal size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)  # location of the face in the resized image
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # find the encoding for the detected face
    print(encodeCurFrame)
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):  # loop through each face location and face encoding
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)   # return true or false
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # calculate the distanec
        print("matches",matches)
        print("faceDis", faceDis)
        matchIndex = np.argmin(faceDis)   # calculate the smallest index
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()   # get the name of the matched face and convert it to the uppercase
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2*2, x1*2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) 
            markAttendance(name)
    cv2.imshow("webcam", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()