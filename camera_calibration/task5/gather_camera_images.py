import cv2
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import sys
sys.path.append('..')

camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
counter = 0
imgNum = 1
imageName = "image{}.jpg"

print(3)
cv2.waitKey(1000)
print(2)
cv2.waitKey(1000)
print(1)
cv2.waitKey(1000)


while True:
    ret, image = camera.read()
    cv2.imshow("Cam 0", image)

    if not(counter % 30):
        cv2.imwrite("camera_images/" + imageName.format(imgNum),  image)
        print("Pew")
        imgNum += 1
        if imgNum > 40:
            break

    cv2.waitKey(30)
    counter += 1

camera.release()
cv2.destroyAllWindows()
