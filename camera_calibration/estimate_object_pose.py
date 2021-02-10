import numpy as np
import cv2
import scipy.io as sio

data = sio.loadmat("intrinsic_data.mat")
cameraMatrix = data["cameraMatrix"]
distortion = data["distortion"]

pixelPoints = np.loadtxt("2D_points.txt")
objectPoints = np.loadtxt("3D_points.txt")

ret, RVec, T = cv2.solvePnP(objectPoints, pixelPoints, cameraMatrix, distortion)

R, _ = cv2.Rodrigues(RVec)

print("R: ", R)
print("Euler Angles: ", RVec)
print("Translation: ", T)
