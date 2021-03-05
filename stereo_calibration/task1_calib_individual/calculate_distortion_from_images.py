import cv2
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import os

loadpath = "../images/test/L/"
sampleImage = cv2.imread(loadpath + "L1.png")
width = sampleImage.shape[1]
height = sampleImage.shape[0]
chessXPoints = 10
chessYPoints = 7


def get_chessboard_subpix(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,
                                             (chessXPoints, chessYPoints),
                                             None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                    0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        # cornerImg = cv2.drawChessboardCorners(sampleImage, (chessXPoints,chessYPoints), subCorners, ret)

    return ret, corners


def calibrate(loadpath, savename="intrinsic_parameters.mat"):

    points = np.zeros((chessXPoints * chessYPoints, 3), np.float32)
    points[:, :2] = 3.88 * np.mgrid[:chessXPoints, :chessYPoints].T.reshape(
        -1, 2)

    objectPoints = []
    imagePoints = []

    # loop = tqdm(total=40, position=0, leave=False)

    for filename in tqdm(os.listdir(loadpath)):
        image = cv2.imread(loadpath + filename)

        ret, corners = get_chessboard_subpix(image)

        if ret:
            objectPoints.append(points)
            imagePoints.append(corners)

        # loop.update(1)

    ret, cameraMatrix, distortion, rotations, translations = cv2.calibrateCamera(
        objectPoints, imagePoints, (width, height), None, None)

    fs = cameraMatrix[0, 0]
    s = 135
    focalLength = fs / s

    print("Focal length is {:.4f} mm".format(focalLength))

    np.set_printoptions(precision=4, suppress=True)
    print("Camera Matrix: {}".format(cameraMatrix))
    print("Distortion: {}".format(distortion))

    data = {"cameraMatrix": cameraMatrix, "distortion": distortion}
    sio.savemat(savename, data)
    print("Parameters saved to file")


loadpath = "../images/test/L/"
savename = "intrinsic_parameters_testL.mat"

calibrate(loadpath, savename)

loadpath = "../images/test/R/"
savename = "intrinsic_parameters_testR.mat"

calibrate(loadpath, savename)
