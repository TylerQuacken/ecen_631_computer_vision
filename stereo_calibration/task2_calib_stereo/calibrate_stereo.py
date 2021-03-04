import cv2
import numpy as np
from tqdm import tqdm
import os
import scipy.io as sio
from IPython import embed

loadpath = "../images/test/"
Ldir = "SL/L"
Rdir = "SR/R"
fileName = "{}.png"
sampleImage = cv2.imread(loadpath + Ldir + "1.png")
width = sampleImage.shape[1]
height = sampleImage.shape[0]
chessXPoints = 10
chessYPoints = 7

dataL = sio.loadmat("../intrinsic_parameters_testL.mat")
dataR = sio.loadmat("../intrinsic_parameters_testR.mat")
cameraL = np.float32(dataL["cameraMatrix"])
cameraR = np.float32(dataR["cameraMatrix"])
distortionL = np.float32(dataL["distortion"])
distortionR = np.float32(dataR["distortion"])


def find_chessboard_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gray = image
    ret, corners = cv2.findChessboardCorners(gray,
                                             (chessXPoints, chessYPoints),
                                             None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000,
                    0.001)
        corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

    return ret, corners


def calibrate(loadpath,
              Ldir,
              Rdir,
              fileName,
              savename="stereo_parameters.mat"):

    points = np.zeros((chessXPoints * chessYPoints, 3), np.float32)
    points[:, :2] = 3.88 * np.mgrid[:chessXPoints, :chessYPoints].T.reshape(
        -1, 2)

    objectPoints = []
    imagePointsL = []
    imagePointsR = []

    # loop = tqdm(total=40, position=0, leave=False)

    for fileNum in tqdm(range(1, 43)):
        imageL = cv2.imread(loadpath + Ldir + fileName.format(fileNum))
        imageR = cv2.imread(loadpath + Rdir + fileName.format(fileNum))

        if imageL is not None and imageR is not None:

            retL, cornersL = find_chessboard_corners(imageL)
            retR, cornersR = find_chessboard_corners(imageR)

            if retL and retR:
                objectPoints.append(points)
                imagePointsL.append(cornersL)
                imagePointsR.append(cornersR)

    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objectPoints,
        imagePointsL,
        imagePointsR,
        cv2.UMat(cameraL),
        distortionL,
        cv2.UMat(cameraR),
        distortionR, (width, height),
        flags=cv2.CALIB_FIX_INTRINSIC)

    R = cv2.UMat.get(R)
    T = cv2.UMat.get(T)
    E = cv2.UMat.get(E)
    F = cv2.UMat.get(F)
    np.set_printoptions(precision=4, suppress=True)
    print("Rotation: {}".format(R))
    print("Translation: {}".format(T))
    print("Essential: {}".format(E))
    print("Fundamental: {}".format(F))

    data = {"R": R, "T": T, "E": E, "F": F}
    sio.savemat(savename, data)
    print("Parameters saved to file")


calibrate(loadpath, Ldir, Rdir, fileName)
