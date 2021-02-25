import numpy as np
import cv2


class Object():
    def __init__(self):
        self.location = np.zeros([3])
        self.mass = 1.0
        self.velocity = np.zeros([3])
        self.xRange = np.array([0., 11.])
        self.yRange = np.array([-8., 0.])
        self.RVec = []

    def set_ground(self, RVec):
        self.groundR = RVec

    def update(self, RVec):
        pass

    def get_points(self):
        # cube points
        points = np.zeros([8, 4, 1])
        points[0, :, 0] = [0, 0, 0, 1]
        points[1, :, 0] = [0, 2, 0, 1]
        points[2, :, 0] = [2, 2, 0, 1]
        points[3, :, 0] = [2, 0, 0, 1]
        points[4, :, 0] = [0, 0, 1, 1]
        points[5, :, 0] = [0, 2, 1, 1]
        points[6, :, 0] = [2, 2, 1, 1]
        points[7, :, 0] = [2, 0, 1, 1]
        points[:, 0:3, 0] = points[:, 0:3, 0] + self.location

        return points[:, 0:3, 0].T

    def draw_object_on_image(self, image, camera, RVec, T):
        projPoints, _ = cv2.projectPoints(self.get_points(), RVec, T,
                                          camera.cameraMatrix,
                                          camera.distortion)
        imgPoints = projPoints.squeeze().astype('int')

        for i in range(len(imgPoints)):
            cv2.circle(image, tuple(imgPoints[i, :]), 10, (0, 0, 255), -1)

        return image
