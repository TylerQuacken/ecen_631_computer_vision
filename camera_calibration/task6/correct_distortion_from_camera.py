import cv2
import numpy as np
import scipy.io as sio


data = sio.loadmat("intrinsic_data_logitech.mat")
cameraMatrix = data["cameraMatrix"]
distortion = data["distortion"]

loadpath = ""

close = cv2.imread(loadpath + "image7.jpg")
width = close.shape[1]
height = close.shape[0]

newMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortion, (width, height), 1, (width, height))

def undistort_image(image):
    fixed = cv2.undistort(image, cameraMatrix, distortion, None)

    x, y, w, h = roi
    fixed2 = fixed[y:y+h, x:x+w]
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayFixed = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(grayImage, grayFixed)
    return diff


closeDiff = undistort_image(close)

cv2.imwrite('image7_difference.jpg', closeDiff)
