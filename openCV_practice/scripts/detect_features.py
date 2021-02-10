import cv2
import numpy as np
from skimage.metrics import structural_similarity
import imutils
from IPython import embed

camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
vout = cv2.VideoWriter('./video.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))

inputCounter = 0
methods = ["binarized", "canny", "corners", "lines", "difference"]
numMethods = len(methods)
ret0, frame = camera.read()

while True:
    prevFrame = frame
    ret0, frame = camera.read()

    if inputCounter == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retProcessed, processedImage = cv2.threshold(gray,110,255,cv2.THRESH_BINARY)
        processedImage = cv2.cvtColor(processedImage, cv2.COLOR_GRAY2BGR)

    elif inputCounter == 2:
        processedImage = cv2.Canny(frame, 100, 200)
        processedImage = cv2.cvtColor(processedImage, cv2.COLOR_GRAY2BGR)

    elif inputCounter == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        harris = cv2.cornerHarris(gray, 2, 3, 0.04)
        dilated = cv2.dilate(harris, None)
        ret, threshold = cv2.threshold(dilated, 0.01*dilated.max(),255,0)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(threshold))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)
        corners = np.int0(corners)
        
        blotSize = 6
        y = np.clip(corners[:,1], 0, height)
        y = np.tile(y, [blotSize*4+1,1])
        x = np.clip(corners[:,0], 0, width)
        x = np.tile(x, [blotSize*4+1,1])
        for i in range(blotSize):
            y[i,:] += i
            y[i+blotSize,:] -= i
            x[i+2*blotSize,:] += i
            x[i+3*blotSize,:] -= i

        y = y.reshape(-1)
        x = x.reshape(-1)
        y = np.clip(y, 0, height-1)
        x = np.clip(x, 0, width-1)
        frame[y, x] = np.array([0, 0, 255])
        processedImage = frame

    elif inputCounter == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        minLineLength = 1000
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

        for i in range(len(lines)):
            x1 = lines[i,0,0]
            y1 = lines[i,0,1]
            x2 = lines[i,0,2]
            y2 = lines[i,0,3]
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

        processedImage = frame

    elif inputCounter == 5:
        grayA = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

        # (score, difference) = structural_similarity(grayA, grayB, full=True)
        # difference = (difference * 255).astype("uint8")
        difference = cv2.absdiff(frame, prevFrame)
        processedImage = difference
        

    else:
        processedImage = frame
        retProcessed = ret0
        
    cv2.imshow("Cam 0", processedImage)
    vout.write(processedImage)
    
    if cv2.waitKey(1) & 0xFF == ord(' '):
        inputCounter += 1

        if inputCounter > numMethods:
            break
        else:
            print(methods[inputCounter-1])

# When everything is done, release the capture
cv2.imwrite('./Cam0.jpg', processedImage)
print("Wrote video to file")
camera.release()
cv2.destroyAllWindows()
