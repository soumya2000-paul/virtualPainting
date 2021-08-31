import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


brushThickness = 15
eraserThickness = 100

folderPath = "khokon"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
khokon = overlayList[0]
drawColor = (255, 255, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon = 0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    #import images
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #Find handlandmark
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:
       #print(lmList)

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #check which finger are up
        fingers = detector.fingersUp()
        # print(fingers)

        #If selection mode 2 finger up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            if y1 < 125:
                if 200 < x1 < 350:
                    khokon = overlayList[0]
                    xp, yp = 0, 0
                    drawColor = (255, 255, 255)
                elif 380 < x1 < 570:
                    khokon = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 600 < x1 < 750:
                    khokon = overlayList[2]
                    drawColor = (255, 255, 0)
                elif 800 < x1 < 950:
                    drawColor = (255, 153, 255)
                elif 1050 < x1 < 1200:
                    khokon = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            #if drawing mode 3 finger up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing MOde")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    #Setting the image
    img[0:120, 0:1212] = khokon
    ##img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5,0)

    cv2.imshow('Koushik', img)
    cv2.imshow('Dibas', imgCanvas)
    cv2.imshow('Prasen ', imgInv)
    cv2.waitKey(1)
