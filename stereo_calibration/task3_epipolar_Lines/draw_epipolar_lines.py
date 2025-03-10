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

loadpath = "../images/test/"
Ldir = "SL/L"
Rdir = "SR/R"
imageLOrig = cv2.imread(loadpath + Ldir + "1.png")
imageROrig = cv2.imread(loadpath + Rdir + "1.png")
imageL = imageLOrig
imageR = imageROrig
width = imageL.shape[1]
height = imageL.shape[0]

imageL = cv2.undistort(imageL, cameraL, distortionL)
imageR = cv2.undistort(imageR, cameraR, distortionR)

pointsL = [(251, 161), (429, 220), (241, 414)]
pointsR = [(297, 161), (91, 251), (458, 301)]
pointsL = np.array([[251, 161], [429, 220], [241, 414]])
pointsR = np.array([[297, 161], [91, 251], [458, 301]])

for i in range(3):
    cv2.circle(imageL, tuple(pointsL[i]), 10, (0, 0, 255), 2)
    cv2.circle(imageR, tuple(pointsR[i]), 10, (0, 0, 255), 2)

linesL = cv2.computeCorrespondEpilines(pointsR, 2, F)
linesR = cv2.computeCorrespondEpilines(pointsL, 1, F)


def drawEpipolar(image, lines):
    for i in range(3):
        r = lines[i, 0, :]
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [width, -(r[2] + r[0] * width) / r[1]])
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1)


drawEpipolar(imageL, linesL)
drawEpipolar(imageR, linesR)
cv2.imwrite("left_points.jpg", imageL)
cv2.imwrite("right_points.jpg", imageR)
cv2.imshow("Left", imageL)
cv2.imshow("Right", imageR)

cv2.waitKey()
