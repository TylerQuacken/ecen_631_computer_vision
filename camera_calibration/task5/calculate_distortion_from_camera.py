import cv2
import numpy as np
from tqdm import tqdm
import scipy.io as sio

loadpath = "camera_images/"
sampleImage = cv2.imread(loadpath + "image1.jpg")
width = sampleImage.shape[1]
height = sampleImage.shape[0]
chessXPoints = 9
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


points = np.zeros((chessXPoints * chessYPoints, 3), np.float32)
points[:, :2] = 24.5 * np.mgrid[:chessXPoints, :chessYPoints].T.reshape(-1, 2)

objectPoints = []
imagePoints = []

loop = tqdm(total=40, position=0, leave=False)

for i in range(1, 41):
    filename = "image{}.jpg".format(i)
    image = cv2.imread(loadpath + filename)

    ret, corners = get_chessboard_subpix(image)

    if ret:
        objectPoints.append(points)
        imagePoints.append(corners)

    loop.update(1)

ret, cameraMatrix, distortion, rotations, translations = cv2.calibrateCamera(
    objectPoints, imagePoints, (width, height), None, None)

fs = cameraMatrix[0, 0]
s = 135
focalLength = fs / s

print("Focal length is {} mm".format(focalLength))

print("Camera Matrix: {}".format(cameraMatrix))
print("Distortion: {}".format(distortion))

data = {"cameraMatrix": cameraMatrix, "distortion": distortion}
sio.savemat("intrinsic_data_logitech.mat", data)
print("Parameters saved to file")

# cv2.imshow('Corners', cornerImg)
# cv2.waitKey(10000)
# cv2.imwrite('./Chessboard_Corners.jpg', cornerImg)

cv2.destroyAllWindows()
