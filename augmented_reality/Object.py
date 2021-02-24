import numpy as np


class Object():
    def __init__(self):
        self.location = np.zeros([3])

    def update(self):
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
