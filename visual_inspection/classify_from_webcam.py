import cv2
from face_utils import FaceClassifier, crop_faces

camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
vout = cv2.VideoWriter('./video.avi', cv2.VideoWriter_fourcc(*'XVID'), 25,
                       (width, height))
faceClassifier = FaceClassifier()
prototxtPath = "deploy.prototxt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
confidenceMin = 0.5

while True:
    ret0, frame = camera.read()

    image = faceClassifier.classify_image(frame)

    cv2.imshow("Cam 0", image)
    vout.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cv2.imwrite('./Cam0.jpg', frame)
camera.release()
cv2.destroyAllWindows()
