import cv2
import numpy as np
from IPython import embed

loadPath = "../data/Baseball Practice Images/"

rParams = {"lowThresh": 8}
lParams = {"lowThresh": 3}
paramsCam = {"R": rParams,
          "L": lParams}


sampleImage = cv2.imread(loadPath + "1L05.jpg")
width = sampleImage.shape[1]
height = sampleImage.shape[0]
vout = cv2.VideoWriter('./video.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))

def locate_ball(img, prevImg, params):
    diffImg = cv2.absdiff(img, prevImg)
    _, thresholdImg = cv2.threshold(diffImg, params["lowThresh"], 255, cv2.THRESH_BINARY)

    mask = cv2.erode(thresholdImg, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    return mask

def process_image(img, mask):
    img[mask>0] = [0,0,254]
    return img

for cam in ("R", "L"):
    prevImg = cv2.imread(loadPath + "1{}05.jpg".format(cam))
    prevImgGray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)

    for i in range(5,41):

        fileName = "1" + cam + str(i).zfill(2) + ".jpg"
        img = cv2.imread(loadPath+fileName)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = locate_ball(imgGray, prevImgGray, paramsCam[cam])

        processedImg = process_image(img, mask)

        cv2.imshow("Feed", processedImg)
        vout.write(processedImg)
        key = cv2.waitKey(75)

cv2.destroyAllWindows()
