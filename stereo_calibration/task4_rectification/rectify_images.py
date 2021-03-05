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
imageL = cv2.imread(loadpath + Ldir + "1.png")
imageR = cv2.imread(loadpath + Rdir + "1.png")
width = imageL.shape[1]
height = imageL.shape[0]

imageL = cv2.undistort(imageL, cameraL, distortionL)
imageR = cv2.undistort(imageR, cameraR, distortionR)

cv2.imwrite("left_points_before.jpg", imageL)
cv2.imwrite("right_points_before.jpg", imageR)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraL, distortionL, cameraR,
                                            distortionR, (width, height), R, T)

map1L, map2L = cv2.initUndistortRectifyMap(cameraL, distortionL, R1, P1,
                                           (width, height), cv2.CV_32FC1)
map1R, map2R = cv2.initUndistortRectifyMap(cameraR, distortionR, R2, P2,
                                           (width, height), cv2.CV_32FC1)

imageLRect = cv2.remap(imageL, map1L, map2L, cv2.INTER_LINEAR)
imageRRect = cv2.remap(imageR, map1R, map2R, cv2.INTER_LINEAR)

numLines = 10
lineSpacing = (height - 40) // numLines
h = 20

for i in range(numLines):
    cv2.line(imageLRect, (0, h), (width, h), (0, 255, 0), 1)
    cv2.line(imageRRect, (0, h), (width, h), (0, 255, 0), 1)
    h += lineSpacing

cv2.imwrite("left_points_after.jpg", imageLRect)
cv2.imwrite("right_points_after.jpg", imageRRect)
cv2.imshow("AfterL", imageLRect)
cv2.imshow("AfterR", imageRRect)

grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
grayLRect = cv2.cvtColor(imageLRect, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)
grayRRect = cv2.cvtColor(imageRRect, cv2.COLOR_BGR2GRAY)

diffL = cv2.absdiff(grayLRect, grayL)
diffR = cv2.absdiff(grayRRect, grayR)

cv2.imwrite("diffL.jpg", diffL)
cv2.imwrite("diffR.jpg", diffR)

cv2.waitKey(10000)
