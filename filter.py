import numpy as np
import dlib
import cv2

def empty(a):
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR", 640, 240)
cv2.createTrackbar("Blue", 'BGR', 0, 255, empty)
cv2.createTrackbar("Green", 'BGR', 0, 255, empty)
cv2.createTrackbar("Red", 'BGR', 0, 255, empty)
def createBox(img, points, scale, masked=False, cropped = True):
    if masked:   # if masked, return the black color around the lip
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))  
        img = cv2.bitwise_and(img, mask)
    if cropped:   # return the bounding box around the lip
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
        return imgCrop
    else:
        return mask  # return black color with white lip
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"Desktop/image/facial-landmarks-recognition\shape_predictor_68_face_landmarks.dat")
while True:
    if cap: 
        success, img = cap.read()
        img = cv2.flip(img, 1)
    else: 
        img = cv2.imread('Desktop/image/New folder/images_and_video/Elon_Musk_1.jpg')
    img = cv2.resize(img, (0, 0), None, 1, 1)

    imgOriginal = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    myPoints = []
    for face in faces:
        x1, y1 = face.left(), face.top()   # the method gives you the left points
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(imgGray, face)
        for n in range(68):  # loop over 68 landmarks
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])
            # cv2.circle(imgOriginal, (x, y), 1, (50, 50, 255), cv2.FILLED)
            # cv2.putText(imgOriginal, str(n), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1)
        myPoints = np.array(myPoints)
        # imgLeftEye = createBox(img, myPoints[36:42])
        imgLips = createBox(img, myPoints[48:61], 3, True, False)
        imgColorLips = np.zeros_like(imgLips)
        b = cv2.getTrackbarPos('Blue', 'BGR')
        g = cv2.getTrackbarPos('Green', 'BGR')
        r = cv2.getTrackbarPos('Red', 'BGR')
        imgColorLips[:] = b, g, r   # get the lip color
        imgColorLips = cv2.bitwise_and(imgColorLips, imgLips)
        imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)   # helps smooth the image and reduce noise
        imgOriginalGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)  
        imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)
        imgColorLips = cv2.addWeighted(imgOriginalGray, 1, imgColorLips, 0.4, 0)  
        # blends the 2 images together by taking a weighted sum of them
        # blends 40% of the colored and blurred lips with the 1005 of the original image
        cv2.imshow('BGR', imgColorLips)
    cv2.imshow("Orginal", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()