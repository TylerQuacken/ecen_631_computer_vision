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
arucoPoints = np.array([[0, 0], [7.25, 0], [-0.06, -4.875], [17.1250, -4.8]])


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


def find_aruco(image, numNeeded=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cornersDetected, ids, rejected = cv2.aruco.detectMarkers(
        gray, arucoDict, parameters=arucoParams)

    # embed()
    corners = []
    idDetected = []

    if len(cornersDetected) < numNeeded:
        ret = False
        print(len(cornersDetected))
    else:
        for i in range(4):
            index = np.where(ids == i)[0]
            if index.size == 0:
                index = None
            else:
                index = index[0]
                corners.append(cornersDetected[index])
                idDetected.append(ids[index])
        ret = True

    if numNeeded == 1 and ret and idDetected[0] != 0:
        ret = False

    return ret, corners, idDetected


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
    yVec = yVec / np.linalg.norm(yVec)

    R = np.zeros([3, 3])
    R[:, 0] = xVec
    R[:, 1] = yVec
    R[:, 2] = zVec

    RVec, _ = cv2.Rodrigues(R)
    T = TVecs[0, :]

    for i in range(TVecs.shape[0]):
        cv2.aruco.drawAxis(image, camera.cameraMatrix, camera.distortion,
                           RVecAruco[i, :, :], TVecs[i, :], 0.7)
    return RVec, T


def get_R_T_from_single_aruco(corners):
    RVecAruco, TVecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, 2, camera.cameraMatrix, camera.distortion)

    T = TVecs[0, 0, :]
    RVec = RVecAruco[0, 0, :]
    return RVec, T


def get_homography_from_single_aruco(corners):
    singleArucoPoints = np.array([[[1, -1], [-1, -1], [-1, 1], [1, 1]]])
    graphy, _ = cv2.findHomography(singleArucoPoints, corners[0], cv2.RANSAC)
    return graphy


while True:
    # Calibrate ground
    image = camera.get_image()
    cv2.imshow("camera", image)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        calibImage = image
        break

ret, corners, ids = find_aruco(calibImage, numNeeded=1)
# RVec, T = get_R_T_from_aruco(corners)
RVec, T = get_R_T_from_single_aruco(corners)
# ret, corners = find_chessboard_corners(image)
# ret, RVec, T = cv2.solvePnP(chessPoints, corners, camera.cameraMatrix,
#                             camera.distortion)
object3d.set_ground(RVec)

lastUpdate = time.time()

while True:

    image = camera.get_image()
    imageUndist = camera.get_image_undistort()
    ret, corners, ids = find_aruco(image, numNeeded=1)
    # ret, corners = find_chessboard_corners(image)

    if ret:
        # ret, RVec, T = cv2.solvePnP(chessPoints, corners, camera.cameraMatrix,
        #                             camera.distortion)
        # RVec, T = get_R_T_from_aruco(corners)
        RVec, T = get_R_T_from_single_aruco(corners)
        # graphy = get_homography_from_single_aruco(corners)

        imageUndist = object3d.draw_object_on_image(imageUndist, camera, RVec,
                                                    T)

        dt = lastUpdate - time.time()
        lastUpdate = time.time()
        object3d.update(RVec, dt)

    flipped = np.flip(imageUndist, axis=1)
    cv2.imshow("camera", flipped)

    cv2.waitKey(30)

camera.release_camera()

cv2.destroyAllWindows()
