import cv2
import mediapipe as mp
import time
import math


class HandDetection:

    def __init__(self, mode=False, max_hands=2, detect_conf=0.5, track_conf=0.5):  # all these are the default values
        self.mode = mode
        self.max_hands = max_hands
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detect_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

# This method is for detecting the hands
    def findHands(self, img, draw=True):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            # Iterating over all hands
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # Adding the "connections" argument helps to join the landmarks using lines otherwise only the points
                    # will get displayed
                    self.mpDraw.draw_landmarks(image=img, landmark_list=hand_lms, connections=self.mpHands.HAND_CONNECTIONS)
        return img

# This method provides the coordinates of all the 21 landmarks
    def findPosition(self, img, hand_no=0, draw=True):

        self.lmlist = []
        bbox = []
        xlist = []
        ylist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xlist.append(cx)
                ylist.append(cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), 2)
            xmin, ymin, xmax, ymax = min(xlist), min(ylist), max(xlist), max(ylist)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0]-25, bbox[1]-25), (bbox[2]+25, bbox[3]+25), (255, 0, 0), 3)

        return self.lmlist, bbox

# This method gives the distance between any 2 landmarks
    def findDistance(self, id1, id2, img, draw=True):

        x1, y1 = self.lmlist[id1][1], self.lmlist[id1][2]
        x2, y2 = self.lmlist[id2][1], self.lmlist[id2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Create circles at both tips
        if draw:
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), 3)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), 3)
            # Draw a line joining both tips
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            # Locate center of line and draw circle
            # cv2.circle(img, (cx, cy), 5, (255, 0, 0), 3)
            # calculate length of line
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

# This methods tells which fingers are up or down
    def fingersUp(self):
        fingers = []
        if len(self.lmlist) != 0:
            # For thumb
            if self.lmlist[self.tipIds[0]][1] < self.lmlist[self.tipIds[0] - 1][1]:
                fingers.append(0)
            else:
                fingers.append(1)
            # For the four fingers
            for i in range(1, 5):

                if self.lmlist[self.tipIds[i]][2] < self.lmlist[self.tipIds[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

def main():
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetection()

    while True:
        success, img = cap.read()
        img = detector.findHands(img=img)
        lmlist = detector.findPosition(img=img)
        if len(lmlist) != 0:
            print(lmlist[4])
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img=img, text=str(int(fps)), org=(20, 100), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=3,
                    color=(255, 0, 255), thickness=3)
        cv2.imshow("image", img)
        cv2.waitKey(5)


if __name__ == "__main__":
    main()