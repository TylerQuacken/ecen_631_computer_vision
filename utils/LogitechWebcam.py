import cv2
import numpy as np
import scipy.io as sio


class LogitechWebcam():
    def __init__(self):
        data = sio.loadmat(
            "/home/tylerquacken/classes/ecen_631_computer_vision/utils/intrinsic_data_logitech.mat"
        )
        self.cameraMatrix = data["cameraMatrix"]
        self.distortion = data["distortion"]

        self.camera = cv2.VideoCapture(0)

        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_image(self):
        ret, image = self.camera.read()
        return image

    def get_image_undistort(self):
        image = self.get_image()
        fixed = cv2.undistort(image, self.cameraMatrix, self.distortion, None)
        return fixed

    def release_camera(self):
        self.camera.release()
