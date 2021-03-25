import numpy as np
import cv2
import sys
sys.path.append('..')
from utils.LogitechWebcam import LogitechWebcam
from IPython import embed

if __name__ == '__main__':
    refFileName = "./images/left_screen.png"
    refImage = cv2.imread(refFileName)
    subImage = cv2.imread("./images/sub_image.jpg")
    gray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    keyPoints, descriptors = orb.detectAndCompute(gray, None)

    camera = LogitechWebcam()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        frame = camera.get_image()
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameKeyPoints, frameDescriptors = orb.detectAndCompute(
            frameGray, None)

        matches = matcher.match(descriptors, frameDescriptors)

        matches = sorted(matches, key=lambda x: x.distance)
        goodMatch = matches[:10]

        # goodMatch = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         goodMatch.append(m)

        minMatches = 9

        if len(goodMatch) > minMatches:
            srcPts = np.float32([keyPoints[m.queryIdx].pt
                                 for m in goodMatch]).reshape(-1, 1, 2)
            dstPts = np.float32([keyPoints[m.trainIdx].pt
                                 for m in goodMatch]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)
            # matchesMask = mask.ravel().tolist()

            h, w, _ = frame.shape

            warp = cv2.warpPerspective(subImage, M, (h, w))

            cv2.imshow("Warped", warp)

        cv2.imshow("Frame", frame)
        cv2.waitKey(30)
