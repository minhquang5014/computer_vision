import cv2
import numpy as np

def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw = False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # blur the images to reduce noise, 5x5 is the kernel size for blurring
    
    # used for edge detection in opencv by using the Canny detection algorithms
    # binary images where edges are marked with white pixels and non-edges are marked with black pixels
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])  
    
    kernel = np.ones((5, 5))

    # enhance the detected edges, 3 is the number of time dilation is applied
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3) 

    # erode is the morphological operation that erodes away boundaries of objects in binary images 
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny: cv2.imshow('Canny', imgThre)

    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContour = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)    # calculate the perimeter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # an approximated polygonal curve with the specified precision
            bbox = cv2.boundingRect(approx)    # draw the bounding box

            # only contours with a specific number of vertices are added to the finalContour list
            if filter > 0:
                if len(approx) == filter:
                    finalContour.append([len(approx), area, approx, bbox, i])
            else:
                finalContour.append([len(approx), area, approx, bbox, i])

        # sort the list with the descending order
        finalContour = sorted(finalContour, key=lambda x:x[1], reverse=True)

        # if the users set the draw to be true, draw the red contours around the objects
        if draw:
            for con in finalContour:
                cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, finalContour

def reorder(Mypoints):  # reorder the 4 points in the x, y, coordinates
    print(Mypoints.shape)

    # create an array with the same shape as the input array
    # This acts as a placeholder for reordered points
    myPointsNew = np.zeros_like(Mypoints) 

    # reshape to ensure the array is suitable for storing the four points 
    # bottom-left, bottom-right, top-left, top-right
    # 4x2 matrix
    Mypoints = Mypoints.reshape((4, 2))  

    # add is the sum of x, y coordinates for each point
    add = Mypoints.sum(1)

    # the point with the smallest sum is assigned to be the top-left
    myPointsNew[0] = Mypoints[np.argmin(add)]

    # the point with the largest sum is assgined to be the bottom-right
    myPointsNew[3] = Mypoints[np.argmax(add)]

    # the difference between the x and y coordinates
    diff = np.diff(Mypoints, axis=1)

    # the point with the smallest difference is the top-right
    myPointsNew[1] = Mypoints[np.argmin(diff)]

    # the point with the largest difference is the bottom-left
    myPointsNew[2] = Mypoints[np.argmax(diff)]
    return myPointsNew
def warpImg(img, points, w, h):
    print(points)
    print(reorder(points))
    # pt1 = np.float32(points)
    # pt2 = np.float32([0, 0], [w, 0], [0, h], [w, h])
    # matrix = cv2.getPerspectiveTransform(pt1, pt2)
    # imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    # return imgWarp