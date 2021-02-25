import numpy as np
import cv2
import sys
from Object import Object
sys.path.append('..')
from utils.LogitechWebcam import LogitechWebcam
import time
from IPython import embed

camera = LogitechWebcam()
chessXPoints = 9
chessYPoints = 7
chessPoints = np.zeros((chessXPoints * chessYPoints, 3), np.float32)
chessPoints[:, :2] = np.mgrid[:chessXPoints, :chessYPoints].T.reshape(-1, 2)

object3d = Object()
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoPoints = np.array([[0, 0, 0], [7.25, 0, 0], [-0.06, -4.875, 0],
                        [17.1250, -4.8, 0]])


def find_chessboard_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,
                                             (chessXPoints, chessYPoints),
                                             None)
    # cornerImg = image

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,
                    0.01)
        corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)
        # cornerImg = cv2.drawChessboardCorners(image,
        #                                       (chessXPoints, chessYPoints),
        #                                       corners, ret)

    return ret, corners


def find_aruco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cornersDetected, ids, rejected = cv2.aruco.detectMarkers(
        gray, arucoDict, parameters=arucoParams)

    if len(cornersDetected) < 4:
        ret = False
        corners = []
        print(len(cornersDetected))
    else:
        corners = []
        for i in range(4):
            index = np.squeeze(np.where(ids == i))[0]
            corners.append(cornersDetected[index])
        ret = True

    return ret, corners, ids


def get_R_T_from_aruco(corners):
    RVecAruco, TVecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, 2, camera.cameraMatrix, camera.distortion)

    TVecs = TVecs.squeeze()
    R01 = TVecs[1, :] - TVecs[0, :]
    R03 = TVecs[3, :] - TVecs[0, :]
    zVec = np.cross(R03, R01)
    zVec = zVec / np.linalg.norm(zVec)
    xVec = R01 / np.linalg.norm(R01)
    yVec = np.cross(zVec, xVec)

    R = np.zeros([3, 3])
    R[:, 0] = xVec
    R[:, 1] = yVec
    R[:, 2] = zVec

    RVec, _ = cv2.Rodrigues(R)
    T = TVecs[0, :]

    # for i in range(TVecs.shape[0]):
    #     cv2.aruco.drawAxis(image, camera.cameraMatrix, camera.distortion,
    #                        RVecAruco[i, :, :], TVecs[i, :], 0.7)
    return RVec, T


while True:

    image = camera.get_image()
    ret, corners, ids = find_aruco(image)
    # ret, corners = find_chessboard_corners(image)

    if ret:
        # ret, RVec, T = cv2.solvePnP(chessPoints, corners, camera.cameraMatrix,
        #                             camera.distortion)
        RVec, T = get_R_T_from_aruco(corners)

        image = object3d.draw_object_on_image(image, camera, RVec, T)

    flipped = np.flip(image, axis=1)
    cv2.imshow("camera", flipped)

    cv2.waitKey(30)

camera.release_camera()

cv2.destroyAllWindows()
