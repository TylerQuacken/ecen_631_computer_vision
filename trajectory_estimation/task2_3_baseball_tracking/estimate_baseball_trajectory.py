import cv2
import numpy as np
from tqdm import tqdm
import os
import scipy.io as sio
import matplotlib.pyplot as plt
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
lName = "L/{:02d}.png"
rName = "R/{:02d}.png"
fileBase = 202
fileLow = 166
fileHigh = 192
nImages = fileHigh - fileLow
baseImageL = cv2.imread(loadpath + lName.format(fileBase))
baseImageR = cv2.imread(loadpath + rName.format(fileBase))
width = baseImageL.shape[1]
height = baseImageL.shape[0]
focusLx = (279, 426)
focusLy = (71, 264)
focusRx = (195, 343)
focusRy = (71, 264)


def focus_image(image, focusX, focusY):
    focused = image[focusY[0]:focusY[1], focusX[0]:focusX[1]]
    widthF = focusX[1] - focusX[0]
    heightF = focusY[1] - focusY[0]

    return focused, widthF, heightF


def unfocus_point(pointF, focusX, focusY):
    offset = np.array([focusX[0], focusY[0]])
    point = pointF + offset
    return point


def unfocus_image(image, imageF, focusX, focusY):
    image[focusX[0]:focusX[1], focusY[0]:focusY[1]] = imageF
    return image


def locate_ball(img, prevImg, lowThresh):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
    diffImg = cv2.absdiff(img, prevImg)
    _, thresholdImg = cv2.threshold(diffImg, lowThresh, 255, cv2.THRESH_BINARY)

    mask = cv2.erode(thresholdImg, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)

    if np.any(mask):
        M = cv2.moments(mask)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        centroid = np.array([cX, cY])

    else:
        centroid = np.zeros([2])
    return centroid


prevImgL = baseImageL
prevImgR = baseImageR
centroidLHist = np.zeros([nImages, 2])
centroidRHist = np.zeros([nImages, 2])
lowThreshL = 20
lowThreshR = 20

for i in range(fileLow, fileHigh):
    imageL = cv2.imread(loadpath + lName.format(i))
    imageR = cv2.imread(loadpath + rName.format(i))

    imageLf, _, _ = focus_image(imageL, focusLx, focusLy)
    prevImgLf, _, _ = focus_image(prevImgL, focusLx, focusLy)
    centroidLf = locate_ball(imageLf, prevImgLf, lowThreshL)
    centroidL = unfocus_point(centroidLf, focusLx, focusLy)

    imageRf, _, _ = focus_image(imageR, focusRx, focusRy)
    prevImgRf, _, _ = focus_image(prevImgR, focusRx, focusRy)
    centroidRf = locate_ball(imageRf, prevImgRf, lowThreshR)
    centroidR = unfocus_point(centroidRf, focusRx, focusRy)

    imageL = cv2.circle(imageL, tuple(centroidL.astype('int')), 10,
                        (0, 0, 255), 2)
    imageR = cv2.circle(imageR, tuple(centroidR.astype('int')), 10,
                        (0, 0, 255), 2)

    imageL = cv2.rectangle(imageL, (focusLx[0], focusLy[0]),
                           (focusLx[1], focusLy[1]), (255, 0, 0))
    imageR = cv2.rectangle(imageR, (focusRx[0], focusRy[0]),
                           (focusRx[1], focusRy[1]), (255, 0, 0))

    # cv2.imshow("mask", imageR)
    # cv2.waitKey(30)

    if np.any((i - fileLow) == np.array([1, 5, 10, 15, 20])):
        cv2.imwrite("output/locatedL{}.png".format(i - fileLow), imageL)
        cv2.imwrite("output/locatedR{}.png".format(i - fileLow), imageR)

    centroidLHist[i - fileLow, :] = centroidL
    centroidRHist[i - fileLow, :] = centroidR

centroidLu = cv2.undistortPoints(centroidLHist,
                                 cameraL,
                                 distortionL,
                                 R=R1,
                                 P=P1).squeeze()
centroidRu = cv2.undistortPoints(centroidRHist,
                                 cameraR,
                                 distortionR,
                                 R=R2,
                                 P=P2).squeeze()

disparity = centroidLu[:, 0] - centroidRu[:, 0]

points2d = np.zeros([nImages, 1, 3])
points2d[:, 0, :2] = centroidLu
points2d[:, 0, 2] = disparity

points3d = cv2.perspectiveTransform(points2d, Q).squeeze()

points3d = points3d + (T / 2).squeeze()
x = points3d[:, 0]
y = points3d[:, 1]
z = points3d[:, 2]

polyY = np.polyfit(z, y, 2)
polyX = np.polyfit(z, x, 2)


def calc_poly(poly, tspan=(0, 475), dt=0.01):
    t = np.arange(tspan[0], tspan[1], dt)
    y = poly[0] * t**2 + poly[1] * t + poly[2]
    return t, y


t, yPred = calc_poly(polyY)
t, xPred = calc_poly(polyX)

plt.ion()

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.axis('equal')
ax1.plot(z, y, '.')
ax1.plot(t, yPred)
ax1.legend(["Measured", "Predicted"])
ax1.set_xlabel("Z (in)")
ax1.set_ylabel("Y (in)")
ax1.set_xlim((475, 0))
ax1.invert_yaxis()

ax2.axis('equal')
ax2.plot(z, x, '.')
ax2.plot(t, xPred)
ax2.legend(["Measured", "Predicted"])
ax2.set_xlabel("Z (in)")
ax2.set_ylabel("X (in)")
ax2.set_xlim((475, 0))

plt.show()
