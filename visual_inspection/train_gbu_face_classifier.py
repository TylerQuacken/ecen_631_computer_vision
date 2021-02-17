from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from FaceDataset import FaceDataset
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from torch.nn.parameter import Parameter
import pdb
from IPython import embed
from NN_Structures import Net

savePath = "./face_recognition_weights.pth"

faceDataset = FaceDataset()
train_loader = DataLoader(faceDataset,
                          batch_size=40,
                          shuffle=True,
                          num_workers=0)

# Instantiate your data loaders
model = Net()
model = model.cuda()
objective = nn.CrossEntropyLoss()

# Instantiate your model and loss and optimizer functions
# objective = torch.nn.MSELoss()  # This was in the video
# objective = nn.CrossEntropyLoss()   # This was from Slack
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Run your training and validation loop and collect stats
# Run your training / validation loops

losses = []
validations = []

# loop = tqdm(total=len(val_loader), position = 0)

epochs = 10

for epoch in range(epochs):

    loop = tqdm(total=len(train_loader), position=0, leave=False)

    for batch, data in enumerate(train_loader):

        image = data["image"].cuda()
        label = data["GBU"].cuda()

        # embed()

        optimizer.zero_grad()
        label_pred = model(image.float())

        # embed()

        loss = objective(label_pred, label.long())

        loss.backward()

        losses.append(loss.item())
        accuracy = 0
        loop.set_description('epoch:{}, loss:{:.4f}'.format(
            epoch, loss.item()))
        loop.update(1)

        optimizer.step()

        if batch % 100 == 0:
            torch.save(model.state_dict(), savePath)

    loop.close()
