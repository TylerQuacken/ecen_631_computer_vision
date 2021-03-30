import numpy as np
import cv2
import sys
sys.path.append('..')
from utils.LogitechWebcam import LogitechWebcam
from IPython import embed

if __name__ == '__main__':
    refFileName = "./images/left_screen_rect.png"
    refImage = cv2.imread(refFileName)
    subImage = cv2.imread("./images/sub_image.jpg")
    gray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    VOut = cv2.VideoWriter('Tak_2_second.avi', fourcc, 3.0, (640, 480))

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray, None)
    # surf = cv2.xfeatures2d_SURF.create(400)
    # keyPoints, descriptors = surf.detectAndCompute(gray, None)

    camera = LogitechWebcam()
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    while True:
        frame = camera.get_image()
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(frameGray, None)
        # frameKeyPoints, frameDescriptors = surf.detectAndCompute(
        #     frameGray, None)

        # matches = matcher.match(des1, des2)

        # matches = sorted(matches, key=lambda x: x.distance)
        # goodMatch = matches[:100]

        matches = matcher.knnMatch(des1, des2, k=2)

        goodMatch = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatch.append(m)

        minMatches = 20
        image_out = frame

        if len(goodMatch) > minMatches:
            srcPts = np.float32([kp1[m.queryIdx].pt
                                 for m in goodMatch]).reshape(-1, 1, 2)
            dstPts = np.float32([kp2[m.trainIdx].pt
                                 for m in goodMatch]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)
            # matchesMask = mask.ravel().tolist()

            h, w, _ = frame.shape

            warp = cv2.warpPerspective(subImage, M, frameGray.shape[0:2][::-1])
            contPix = np.where(((warp[:, :, 0] == 0) + (warp[:, :, 0] == 1) +
                                (warp[:, :, 2] == 0)) == False)

            image_out[contPix[0], contPix[1], :] = warp[contPix[0],
                                                        contPix[1], :]

            # cv2.imshow("Warped", warp)

        cv2.imshow("Frame", image_out)
        VOut.write(image_out)
        if cv2.waitKey(30) != -1:
            break

    VOut.release()
    cv2.destroyAllWindows()
