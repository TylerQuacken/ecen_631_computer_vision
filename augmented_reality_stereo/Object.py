import numpy as np
import cv2
from IPython import embed


class Object():
    def __init__(self):
        self.position = np.zeros([3])
        self.mass = 1.0
        self.velocity = np.zeros([3])
        self.positionMax = np.array([8.5, 0.5, 0.])
        self.positionMin = np.array([-0.5, -6., 0.])
        self.groundR = np.zeros([3, 3])

    def set_ground(self, RVec):
        self.groundR, _ = cv2.Rodrigues(RVec)

    def update(self, RVec, T, puckCamera, dt):
        R, _ = cv2.Rodrigues(RVec)
        gVecWorld = 1.5 * np.array([0., 0., -9.8])
        gVecCamera = self.groundR.T @ gVecWorld
        gVecPlane = R @ gVecCamera

        accel = gVecPlane * np.array([-1., 1., 0.])

        if puckCamera is not None:

            puckPlane = R @ (puckCamera - T) / 25.4
            puckPlane[2] = 0.0

            QP = (puckPlane - self.position)[:2]
            r = np.linalg.norm(QP)
            if r < .5:
                r = 0.5
            # print(puckPlane, self.position)
            # if np.linalg.norm(QP) < 2.25:
            #     print("Hit")
            #     self.velocity -= 1.5 * np.dot(self.velocity, QP)
            G = 30.0
            accel[:2] += G / r**2 * QP / r

        self.velocity += accel * dt
        self.position += self.velocity * dt
        tooHigh = np.where(self.position > self.positionMax)
        self.position[tooHigh] = self.positionMax[tooHigh]
        self.velocity[tooHigh] = -0.5 * self.velocity[tooHigh]
        tooLow = np.where(self.position < self.positionMin)
        self.position[tooLow] = self.positionMin[tooLow]
        self.velocity[tooLow] = -0.5 * self.velocity[tooLow]

    def get_points(self):
        # cube points
        # points = np.zeros([3, 8])
        # points[:, 0] = [-1, -1, 0]
        # points[:, 1] = [-1, 1, 0]
        # points[:, 2] = [1, 1, 0]
        # points[:, 3] = [1, -1, 0]
        # points[:, 4] = [-1, -1, 1]
        # points[:, 5] = [-1, 1, 1]
        # points[:, 6] = [1, 1, 1]
        # points[:, 7] = [1, -1, 1]
        # points = points + np.expand_dims(self.position, axis=1)

        # lines = np.array([[0, 1], [0, 4], [0, 3], [1, 2], [1, 5], [2, 6],
        #                   [2, 3], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7]])

        nPoints = 20
        points = np.zeros([3, nPoints * 2])
        for i in range(nPoints):
            theta = 2 * np.pi * i / nPoints
            points[:, i] = [np.cos(theta), np.sin(theta), 0]
            points[:, i + nPoints] = [np.cos(theta), np.sin(theta), 1]

        points = points + np.expand_dims(self.position, axis=1)

        lines = nPoints

        return points * 25, lines

    def draw_object_on_image(self, image, camera, RVec, T):
        points, lines = self.get_points()
        projPoints, _ = cv2.projectPoints(points, RVec, T, camera.cameraL,
                                          camera.distL)
        imgPoints = np.round_(projPoints.squeeze()).astype('int')

        # for i in range(len(imgPoints)):
        #     cv2.circle(image, tuple(imgPoints[i, :]), 10, (0, 0, 255), -1)

        if type(lines) != int:
            for i in range(len(lines)):
                endPointNums = lines[i, :]
                endPoints = imgPoints[endPointNums, :]
                cv2.line(image, tuple(endPoints[0, :]), tuple(endPoints[1, :]),
                         (0, 0, 255), 2)
        else:
            for i in range(lines):
                iPlus1 = (i + 1) % lines
                cv2.line(image, tuple(imgPoints[i, :]),
                         tuple(imgPoints[iPlus1, :]), (0, 0, 255), 2)

                cv2.line(image, tuple(imgPoints[i + lines, :]),
                         tuple(imgPoints[iPlus1 + lines, :]), (0, 0, 255), 2)

                cv2.line(image, tuple(imgPoints[i, :]),
                         tuple(imgPoints[i + lines, :]), (0, 0, 255), 2)

        return image
