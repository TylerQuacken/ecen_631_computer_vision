import numpy as np
import cv2
import sys
sys.path.append('..')
from utils.LogitechWebcam import LogitechWebcam

camera = LogitechWebcam()
chessXPoints = 9
chessYPoints = 7
chessPoints = np.array(False)


def find_chessboard_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,
                                             (chessXPoints, chessYPoints),
                                             None)
    cornerImg = image

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000,
                    0.001)
        corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)
        cornerImg = cv2.drawChessboardCorners(image,
                                              (chessXPoints, chessYPoints),
                                              corners, ret)

    return ret, corners, cornerImg


while True:

    image = camera.get_image()

    ret, corners, cornerImg = find_chessboard_corners(image)

    # if ret:
    #     ret, RVec, T = cv2.solvePnP(chessPoints, corners, camera.cameraMatrix,
    #                                 camera.distortion)

    #     R = cv2.Rodrigues(RVec)

    flipped = np.flip(cornerImg, axis=1)
    cv2.imshow("camera", flipped)

    cv2.waitKey(15)

camera.release_camera()

cv2.destroyAllWindows()
