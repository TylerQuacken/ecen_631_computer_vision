import cv2
import numpy as np
import scipy.io as sio


data = sio.loadmat("intrinsic_data.mat")
cameraMatrix = data["cameraMatrix"]
distortion = data["distortion"]

loadpath = "images_to_correct/"

close = cv2.imread(loadpath + "Close.jpg")
far = cv2.imread(loadpath + "Far.jpg")
turn = cv2.imread(loadpath + "Turn.jpg")
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
farDiff = undistort_image(far)
turnDiff = undistort_image(turn)

cv2.imwrite('close_difference.jpg', closeDiff)
cv2.imwrite('far_difference.jpg', farDiff)
cv2.imwrite('turn_difference.jpg', turnDiff)
