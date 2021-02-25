import numpy as np
import cv2
from IPython import embed


class Object():
    def __init__(self):
        self.position = np.zeros([3])
        self.mass = 1.0
        self.velocity = np.zeros([3])
        self.positionMax = np.array([0., 4., 0.])
        self.positionMin = np.array([-8., -1., 0.])
        self.groundR = np.zeros([3, 3])

    def set_ground(self, RVec):
        self.groundR, _ = cv2.Rodrigues(RVec)

    def update(self, RVec, dt):
        R, _ = cv2.Rodrigues(RVec)
        gVecWorld = np.array([0., 0., -9.8])
        gVecCamera = self.groundR.T @ gVecWorld
        gVecPlane = R @ gVecCamera

        accel = gVecPlane * np.array([1., 1., 0.])
        self.velocity += accel * dt
        self.position += self.velocity * dt
        tooHigh = np.where(self.position > self.positionMax)
        self.position[tooHigh] = self.positionMax[tooHigh]
        self.velocity[tooHigh] = 0.0
        tooLow = np.where(self.position < self.positionMin)
        self.position[tooLow] = self.positionMin[tooLow]
        self.velocity[tooLow] = 0.0

        print(self.position)

    def get_points(self):
        # cube points
        points = np.zeros([3, 8])
        points[:, 0] = [0, 0, 0]
        points[:, 1] = [0, 2, 0]
        points[:, 2] = [2, 2, 0]
        points[:, 3] = [2, 0, 0]
        points[:, 4] = [0, 0, 1]
        points[:, 5] = [0, 2, 1]
        points[:, 6] = [2, 2, 1]
        points[:, 7] = [2, 0, 1]
        points = points + np.expand_dims(self.position, axis=1)

        lines = np.array([[0, 1], [0, 4], [0, 3], [1, 2], [1, 5], [2, 6],
                          [2, 3], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7]])

        return points, lines

    def draw_object_on_image(self, image, camera, RVec, T):
        points, lines = self.get_points()
        projPoints, _ = cv2.projectPoints(points, RVec, T, camera.cameraMatrix,
                                          camera.distortion)
        imgPoints = np.round_(projPoints.squeeze()).astype('int')

        # for i in range(len(imgPoints)):
        #     cv2.circle(image, tuple(imgPoints[i, :]), 10, (0, 0, 255), -1)

        for i in range(len(lines)):
            endPointNums = lines[i, :]
            endPoints = imgPoints[endPointNums, :]
            cv2.line(image, tuple(endPoints[0, :]), tuple(endPoints[1, :]),
                     (0, 0, 255), 4)

        return image
