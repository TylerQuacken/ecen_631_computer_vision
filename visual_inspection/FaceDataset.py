from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from IPython import embed


class FaceDataset(Dataset):
    def __init__(self, rootDir='./images/all/'):
        self.rootDir = rootDir
        self.numG = 142725
        self.numB = 30817
        self.numU = 3317
        self.length = self.numG + self.numB + self.numU

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        index = idx
        numG = self.numG
        numB = self.numB
        numU = self.numU

        prefix = ["G", "B", "U"]
        gbu = 0

        if index >= numG + numB:
            index -= numG + numB
            gbu = 2

        elif index >= numG:
            index -= numG
            gbu = 1

        fileName = prefix[gbu] + "{:06d}.jpg".format(index)

        image = cv2.imread(self.rootDir + fileName)

        sample = {'image': image, 'GBU': gbu}

        return sample
