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

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidenceMin:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            croppedImage = image[startY:endY, startX:endX, :]
            faces.append(croppedImage)

    return faces


if __name__ == "__main__":
    import numpy as np
    import cv2
    imagePath = "images/mugshot/front/A00147"
    # imagePath = "family.jpg"
    prototxtPath = "deploy.prototxt"
    modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
    confidenceMin = 0.2

    image = cv2.imread(imagePath)

    faces = crop_faces(image, prototxtPath, modelPath, confidenceMin)

    for face in faces:
        cv2.imshow("face", face)
        cv2.waitKey(0)
