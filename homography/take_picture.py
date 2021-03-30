import cv2
import sys
sys.path.append('..')
from utils.LogitechWebcam import LogitechWebcam

cam = LogitechWebcam()
# Initiate STAR detector
orb = cv2.ORB_create()

while 1:
    frame = cam.get_image()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find the keypoints with ORB
    kp = orb.detect(frameGray, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(frameGray, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow("Camera", img2)

    if cv2.waitKey(30) != -1:
        break

cv2.imwrite("left_screen.png", frame)
print("Image written to file")

cam.release_camera()
cv2.destroyAllWindows()
