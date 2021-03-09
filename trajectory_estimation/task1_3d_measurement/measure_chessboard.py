import cv2
import numpy as np
from tqdm import tqdm
import os
import scipy.io as sio
from IPython import embed

dataL = sio.loadmat("../intrinsic_parameters_testL.mat")
dataR = sio.loadmat("../intrinsic_parameters_testR.mat")
cameraL = np.float32(dataL["cameraMatrix"])
cameraR = np.float32(dataR["cameraMatrix"])
distortionL = np.float32(dataL["distortion"])
distortionR = np.float32(dataR["distortion"])

dataS = sio.loadmat("../stereo_parameters.mat")
R = dataS["R"]
T = dataS["T"]
E = dataS["E"]
F = dataS["F"]

dataRect = sio.loadmat("../rectification_params.mat")
R1 = dataRect['R1']
R2 = dataRect['R2']
P1 = dataRect['P1']
P2 = dataRect['P2']
Q = dataRect['Q']

loadpath = "../images/calib/"
imageL = cv2.imread(loadpath + "L4.png")
imageR = cv2.imread(loadpath + "R4.png")
width = imageL.shape[1]
height = imageL.shape[0]
chessXPoints = 10
chessYPoints = 7


def find_chessboard_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gray = image
    ret, corners = cv2.findChessboardCorners(gray,
                                             (chessXPoints, chessYPoints),
                                             None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                    0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    return ret, corners


_, cornersL = find_chessboard_corners(imageL)
_, cornersR = find_chessboard_corners(imageR)

indeces = [0, chessXPoints - 1, -chessXPoints, -1]
cornersL = cornersL[indeces, 0, :]
cornersR = cornersR[indeces, 0, :]

cornersLu = cv2.undistortPoints(cornersL, cameraL, distortionL, R=R1,
                                P=P1).squeeze()
cornersRu = cv2.undistortPoints(cornersR, cameraR, distortionR, R=R2,
                                P=P2).squeeze()

# create vectors of points [x, y, disparity]
disparity = cornersLu[:, 0] - cornersRu[:, 0]
corners = np.zeros([4, 1, 3])
corners[:, 0, :2] = cornersLu
corners[:, 0, 2] = disparity

points = cv2.perspectiveTransform(corners, Q)
print(points.squeeze())

for i in range(4):
    cv2.circle(imageL,
               tuple(cornersL[i, :].astype('int')),
               10, (0, 0, 255),
               thickness=2)
    cv2.circle(imageR,
               tuple(cornersR[i, :].astype('int')),
               10, (0, 0, 255),
               thickness=2)

cv2.imwrite("left_corners.png", imageL)
cv2.imwrite("right_corners.png", imageR)

cv2.imshow("left", imageL)
cv2.imshow("right", imageR)
cv2.waitKey(100000)
cv2.destroyAllWindows()
