import numpy as np
import cv2
import sys
from Object import Object
sys.path.append('..')
from utils.ZedCamera import ZedCamera
import time
from IPython import embed

camera = ZedCamera()
chessXPoints = 9
chessYPoints = 7
chessPoints = np.zeros((chessXPoints * chessYPoints, 3), np.float32)
chessPoints[:, :2] = 24.5 * np.mgrid[:chessXPoints, :chessYPoints].T.reshape(
    -1, 2)

object3d = Object()
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoPoints = 25.4 * np.array([[0, 0], [7.25, 0], [-0.06, -4.875],
                               [17.1250, -4.8]])


def find_aruco(image, numNeeded=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cornersDetected, ids, rejected = cv2.aruco.detectMarkers(
        gray, arucoDict, parameters=arucoParams)
    centroids = np.zeros([4, 2])

    if len(cornersDetected) < numNeeded:
        ret = False
    else:
        for i in range(4):
            index = np.where(ids == i)[0]
            if index.size == 0:
                index = None
            else:
                index = index[0]
                centroids[i, :] = get_centroid_from_corners(
                    cornersDetected[index].squeeze())

        ret = True

    return ret, centroids


def get_centroid_from_corners(corners):
    x0, y0 = tuple(corners[0, :])
    x1, y1 = tuple(corners[1, :])
    x2, y2 = tuple(corners[2, :])
    x3, y3 = tuple(corners[3, :])
    m02 = (y0 - y2) / (x0 - x2)
    m13 = (y1 - y3) / (x1 - x3)
    xc = (m02 * x0 - y0 - m13 * x1 + y1) / (m02 - m13)
    yc = m02 * (xc - x0) + y0
    centroid = np.array([xc, yc])
    return centroid


def get_R_T_from_aruco(centroidsL, centroidsR):
    centroidLu = cv2.undistortPoints(centroidsL,
                                     camera.cameraL,
                                     camera.distL,
                                     R=camera.R1,
                                     P=camera.P1).squeeze()
    centroidRu = cv2.undistortPoints(centroidsR,
                                     camera.cameraR,
                                     camera.distR,
                                     R=camera.R2,
                                     P=camera.P2).squeeze()

    disparity = centroidLu[:, 0] - centroidRu[:, 0]

    points2d = np.zeros([4, 1, 3])
    points2d[:, 0, :2] = centroidLu
    points2d[:, 0, 2] = disparity

    points3d = cv2.perspectiveTransform(points2d, camera.Q).squeeze()

    # points3d = points3d + (camera.T / 2).squeeze()

    points3d = points3d.squeeze()
    R01 = points3d[1, :] - points3d[0, :]
    R03 = points3d[3, :] - points3d[0, :]
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
    T = points3d[0, :]

    return RVec, T


def locate_puck(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower_green = np.array([84, 40, 40]) * np.array(
    #     [1 / 359, 1 / 100, 1 / 100]) * 255
    # upper_green = np.array([150, 100, 100]) * np.array(
    #     [1 / 359, 1 / 100, 1 / 100]) * 255
    lower_green = np.array([35, 100, 10])
    upper_green = np.array([100, 255, 120])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # lower_white = np.array([0, 100, 100])
    # upper_white = np.array([20, 255, 255])

    # mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # mask = mask_green mask_white

    mask = cv2.erode(mask, None, iterations=6)
    mask = cv2.dilate(mask, None, iterations=6)

    if np.any(mask):
        M = cv2.moments(mask)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        centroid = np.array([cX, cY])

    else:
        centroid = None
    return centroid


def puck_to_camera(centroidL, centroidR):
    centroidLu = cv2.undistortPoints(centroidL.astype('float64'),
                                     camera.cameraL,
                                     camera.distL,
                                     R=camera.R1,
                                     P=camera.P1).squeeze()
    centroidRu = cv2.undistortPoints(centroidR.astype('float64'),
                                     camera.cameraR,
                                     camera.distR,
                                     R=camera.R2,
                                     P=camera.P2).squeeze()

    disparity = centroidLu[0] - centroidRu[0]

    point2d = np.zeros([1, 1, 3])
    point2d[0, 0, :2] = centroidLu
    point2d[0, 0, 2] = disparity

    point3d = cv2.perspectiveTransform(point2d, camera.Q).squeeze()

    # points3d = points3d + (camera.T / 2).squeeze()

    return point3d


while True:
    # Calibrate ground
    imageL, imageR = camera.get_images()
    ret, centroids = find_aruco(imageL)
    if ret:
        for i in range(4):
            cv2.circle(imageL, tuple(centroids[i, :].astype('int')), 2,
                       (0, 0, 255), -1)

    cv2.imshow("camera", imageL)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        calibImageL = imageL
        calibImageR = imageR
        break

ret, centroidsL = find_aruco(calibImageL)
ret, centroidsR = find_aruco(calibImageR)
RVec, T = get_R_T_from_aruco(centroidsL, centroidsR)
object3d.set_ground(RVec)

lastUpdate = time.time()

while True:

    imageL, imageR = camera.get_images()
    ret, cornersL = find_aruco(imageL)
    ret1, cornersR = find_aruco(imageR)
    # ret, corners = find_chessboard_corners(image)

    if ret and ret1:
        # ret, RVec, T = cv2.solvePnP(chessPoints, corners, camera.cameraMatrix,
        #                             camera.distortion)
        # RVec, T = get_R_T_from_aruco(corners)
        RVec, T = get_R_T_from_aruco(cornersL, cornersR)
        # graphy = get_homography_from_single_aruco(corners)

        centroidPuckL = locate_puck(imageL)
        centroidPuckR = locate_puck(imageR)
        if (centroidPuckL is not None) and (centroidPuckR is not None):
            puckCamera = puck_to_camera(centroidPuckL, centroidPuckR)
        else:
            puckCamera = None

        imageUndist = object3d.draw_object_on_image(imageL, camera, RVec, T)

        dt = lastUpdate - time.time()
        lastUpdate = time.time()
        object3d.update(RVec, T, puckCamera, dt)

    flipped = np.flip(imageUndist, axis=1)
    cv2.imshow("camera", flipped)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

camera.release_camera()

cv2.destroyAllWindows()
