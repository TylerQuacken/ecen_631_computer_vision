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

loadpath = "../images/baseball_trajectory/"
lName = "L/1L{:02d}"
rName = "R/1R{:02d}"
fileLow = 5
fileHigh = 40
baseImageL = cv2.imread(loadpath + lName.format(fileLow))
baseImageR = cv2.imread(loadpath + rName.format(fileLow))
width = baseImageL.shape[1]
height = baseImageL.shape[0]


def locate_ball(img, prevImg, lowThresh):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
    diffImg = cv2.absdiff(img, prevImg)
    _, thresholdImg = cv2.threshold(diffImg, lowThresh, 255, cv2.THRESH_BINARY)

    mask = cv2.erode(thresholdImg, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)

    M = cv2.moments(mask)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    centroid = np.array([cX, cY])
    return centroid


prevImgL = baseImageL
prevImgR = baseImageR

for i in range(fileLow, fileHigh):
    imageL = cv2.imread(loadpath + lName.format(i))
    imageR = cv2.imread(loadpath + rName.format(i))

    centroidL = locate_ball(imageL, prevImgL)
    centroidR = locate_ball(imageR, prevImgR)

    centroidLu = cv2.undistortPoints(centroidL,
                                     cameraL,
                                     distortionL,
                                     R=R1,
                                     P=P1).squeeze()
    centroidRu = cv2.undistortPoints(centroidR,
                                     cameraR,
                                     distortionR,
                                     R=R2,
                                     P=P2).squeeze()

    disparity = centroidLu[0] - centroidRu[0]

    location = np.zeros([1, 3])
    location[0, :2] = centroidLu
    location[0, 2] = disparity
