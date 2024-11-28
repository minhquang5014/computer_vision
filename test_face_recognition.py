import face_recognition
import cv2
import numpy as np

def preprocess_image(img):    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_equalization = cv2.equalizeHist(gray)
    preprocessed = cv2.cvtColor(img_equalization, cv2.COLOR_GRAY2BGR)
    return preprocessed

img = face_recognition.load_image_file('images_and_video/idol.jpg')
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
preprocess_image(img)
img2 = face_recognition.load_image_file('images_and_video/vuonghacde.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
preprocess_image(img2)

faceLoc = face_recognition.face_locations(img)   # return 4 values
faceEnc = face_recognition.face_encodings(img)
#cv2.rectangle(img, (faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
print(faceLoc)
faceLoc2 = face_recognition.face_locations(img2)  # return 4 values
faceEnc2 = face_recognition.face_encodings(img2)
#cv2.rectangle(img2, (faceLoc2[3], faceLoc2[0]),(faceLoc2[1], faceLoc2[2]), (255, 0, 255), 2)

if faceLoc and faceEnc and faceLoc2 and faceEnc2:
    for top, right, bottom, left in faceLoc:
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 255), 2)
    for top, right, bottom, left, in faceLoc2:
        cv2.rectangle(img2, (left, top), (right, bottom), (255, 0, 255), 2)
    results = face_recognition.compare_faces([faceEnc[0]], faceEnc2[0])
    faceDis = face_recognition.face_distance([faceEnc[0]], faceEnc2[0])
    print(results, faceDis)
    cv2.putText(img2, f'{results} {round(faceDis[0], 2)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
else:
    print("No face detected")

cv2.imshow('me3', img)
cv2.imshow('me4', img2)
cv2.waitKey(0)