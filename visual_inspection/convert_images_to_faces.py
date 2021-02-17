import numpy as np
import cv2
from face_utils import crop_faces
import os
from tqdm import tqdm

loadDirectory = "images/img_align_celeba/"
saveDirectory = "images/good/"
prototxtPath = "deploy.prototxt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
confidenceMin = 0.75
fileCounter = 0
reshapeSize = (100, 150)

path, dirs, files = next(os.walk(loadDirectory))
file_count = 202599

loop = tqdm(total=file_count, position=0, leave=False)

for fileName in os.listdir(loadDirectory):
    image = cv2.imread(loadDirectory + fileName)

    success, faces = crop_faces(image, prototxtPath, modelPath, confidenceMin)

    loop.update(1)

    for face in faces:
        resizedFace = cv2.resize(face, reshapeSize)
        faceEdges = cv2.Canny(resizedFace, 125, 175)
        faceEdges = cv2.cvtColor(faceEdges, cv2.COLOR_GRAY2BGR)
        fileName = "{:06d}.jpg".format(fileCounter)
        cv2.imwrite(saveDirectory + fileName, faceEdges)
        fileCounter += 1

loadDirectory = "images/mugshot/"
saveDirectory = "images/bad/"
fileCounter = 0

# path, dirs, files = next(os.walk(loadDirectory))
file_count = 70008

loop = tqdm(total=file_count, position=0, leave=False)

for fileName in os.listdir(loadDirectory):
    image = cv2.imread(loadDirectory + fileName)

    success, faces = crop_faces(image, prototxtPath, modelPath, confidenceMin)

    loop.update(1)

    for face in faces:
        resizedFace = cv2.resize(face, reshapeSize)
        faceEdges = cv2.Canny(resizedFace, 125, 175)
        faceEdges = cv2.cvtColor(faceEdges, cv2.COLOR_GRAY2BGR)
        fileName = "{:06d}.jpg".format(fileCounter)
        cv2.imwrite(saveDirectory + fileName, faceEdges)
        fileCounter += 1

loadDirectory = "images/asian_girls/"
saveDirectory = "images/ugly/"
fileCounter = 0

# path, dirs, files = next(os.walk(loadDirectory))
file_count = 3318

loop = tqdm(total=file_count, position=0, leave=False)

for fileName in os.listdir(loadDirectory):
    image = cv2.imread(loadDirectory + fileName)

    success, faces = crop_faces(image, prototxtPath, modelPath, confidenceMin)

    loop.update(1)

    for face in faces:
        resizedFace = cv2.resize(face, reshapeSize)
        faceEdges = cv2.Canny(resizedFace, 125, 175)
        faceEdges = cv2.cvtColor(faceEdges, cv2.COLOR_GRAY2BGR)
        fileName = "{:06d}.jpg".format(fileCounter)
        cv2.imwrite(saveDirectory + fileName, faceEdges)
        fileCounter += 1
