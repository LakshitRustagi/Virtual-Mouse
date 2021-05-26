# This works for both hands
import cv2
import HandTrackingModule as htm
import numpy as np
import pyautogui

wCam, hCam = 640, 480
wScr, hScr = pyautogui.size()
frameR = 100
smoothening = 7
px, py = 0, 0
cx, cy = 0, 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetection(detect_conf=0.7, max_hands=1)

while True:
    success, img = cap.read()
    # Find Hand landmarks
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img, draw=False)
    if len(lmlist):
        # Checking which fingers are up
        fingers = detector.fingersUp()
        x1, y1 = lmlist[8][1], lmlist[8][2]
        x2, y2 = lmlist[12][1], lmlist[12][2]
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (0, 255, 0), 2)
        # Moving Mode : Only index finger up
        if fingers[1] and not fingers[2]:
            # Convert coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # Smoothen values
            cx = px + (x3-px)/smoothening
            cy = py + (y3-py)/smoothening
            # Move mouse
            pyautogui.moveTo(wScr-cx, cy)
            cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            px, py = cx, cy
        # Clicking mode : Both fingers Up
        if fingers[1] and fingers[2]:
            # Find distance bw index and thumb
            length, img, coordinates = detector.findDistance(8, 12, img)
            # If distance is less than a threshold then click is detected
            if length < 40:
                cv2.circle(img, (coordinates[4], coordinates[5]), 5, (0, 0, 255), cv2.FILLED)
                pyautogui.leftClick()

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()