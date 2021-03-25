import cv2
import numpy as np
import scipy.io as sio
import os


class ZedCamera():
    def __init__(self):
        thisDir = os.path.dirname(__file__)
        dataDir = os.path.join(thisDir, 'ZED_stereo_params.mat')
        data = sio.loadmat(dataDir)
        self.cameraL = data["cameraL"]
        self.cameraR = data["cameraR"]
        self.distL = data["distL"]
        self.distR = data["distR"]
        self.R1 = data["R1"]
        self.R2 = data["R2"]
        self.P1 = data["P1"]
        self.P2 = data["P2"]
        self.Q = data["Q"]
        self.R = data["R"]
        self.T = data["T"]

        self.camera = cv2.VideoCapture(0)

        self.width = 1280
        self.height = 720
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width * 2)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.mapLeftx, self.mapLefty = cv2.initUndistortRectifyMap(
            self.cameraL, self.distL, self.R1, self.P1,
            (self.width, self.height), cv2.CV_32FC1)
        self.mapRightx, self.mapRighty = cv2.initUndistortRectifyMap(
            self.cameraR, self.distR, self.R2, self.P2,
            (self.width, self.height), cv2.CV_32FC1)

    # def __del__(self):
    #     self.release_camera()

    def get_images(self):
        ret, image = self.camera.read()
        imageRL = np.split(image, 2, axis=1)
        imageL = imageRL[0]
        imageR = imageRL[1]
        return imageL, imageR

    def get_images_undistort(self):
        imageL, imageR = self.get_images()
        leftRect = cv2.remap(imageL,
                             self.mapLeftx,
                             self.mapLefty,
                             interpolation=cv2.INTER_LINEAR)
        rightRect = cv2.remap(imageR,
                              self.mapRightx,
                              self.mapRighty,
                              interpolation=cv2.INTER_LINEAR)
        return leftRect, rightRect

    def release_camera(self):
        self.camera.release()
