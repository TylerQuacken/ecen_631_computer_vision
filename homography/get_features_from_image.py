import numpy as np
import cv2

if __name__ == '__main__':
    fileName = "./images/left_screen_rect.png"
    image = cv2.imread(fileName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    keyPoints, descriptors = orb.detectAndCompute(gray, None)

    for keyPt in keyPoints:
        point = keyPt.pt
        cv2.circle(image, tuple(map(int, point)), 3, (255, 0, 0), -1)

    cv2.imshow("points", image)
    cv2.imwrite("./points.png", image)
    cv2.waitKey(0)
