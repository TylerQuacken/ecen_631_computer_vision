import numpy as np
import cv2
import torch
from NN_Structures import Net


def crop_faces(image, prototxtPath, modelPath, confidenceMin):
    """
    ARGUMENTS
    ---------
    image - A numpy array

    RETURNS
    -------
    faces - A tuple of numpy arrays consisting of cropped faces

    """

    net = cv2.dnn.readNetFromCaffe(prototxtPath, modelPath)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    success = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidenceMin:
            success == True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            croppedImage = image[startY:endY, startX:endX, :]
            faces.append(croppedImage)

    return success, faces


class FaceClassifier():
    def __init__(self,
                 confidenceMin=0.5,
                 weightPath="face_recognition_weights.pth",
                 prototxtPath="deploy.prototxt",
                 caffeModelPath="res10_300x300_ssd_iter_140000.caffemodel"):
        self.model = Net()
        self.model.load_state_dict(torch.load(weightPath))
        self.reshapeSize = (100, 150)
        self.prototxtPath = prototxtPath
        self.modelPath = caffeModelPath
        self.classifications = ["Smiley", "Frowny", "Extra Special"]
        self.confidenceMin = confidenceMin

    def classify_image(self, image):
        net = cv2.dnn.readNetFromCaffe(self.prototxtPath, self.modelPath)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        success = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            from IPython import embed
            embed()

            if confidence > self.confidenceMin:
                success == True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX, :]
                classificationNum = self.classify_face(face)
                classification = self.classifications[classificationNum]

                # draw the bounding box of the face along with the associated probability
                text = classification
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)

        return image

    def classify_face(self, face):
        resizedFace = cv2.resize(face, self.reshapeSize)
        faceEdges = cv2.Canny(resizedFace, 125, 175)
        faceEdges = cv2.cvtColor(faceEdges, cv2.COLOR_GRAY2BGR)
        image_t = torch.from_numpy(np.expand_dims(faceEdges, 0)).float()
        output = self.model(image_t)
        classification = torch.argmax(output)
        return classification


if __name__ == "__main__":
    import numpy as np
    import cv2
    imagePath = "images/mugshot/A00147"
    # imagePath = "family.jpg"
    prototxtPath = "deploy.prototxt"
    modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
    confidenceMin = 0.5

    image = cv2.imread(imagePath)

    success, faces = crop_faces(image, prototxtPath, modelPath, confidenceMin)

    for face in faces:
        cv2.imshow("face", face)
        cv2.waitKey(0)
