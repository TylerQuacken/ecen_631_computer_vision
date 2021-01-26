import cv2 as cv 

camera = cv.VideoCapture(0)
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
vout = cv.VideoWriter('./video.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))

inputCounter = 0
methods = ["binarized", "canny", "corners", "lines", "difference"]
numMethods = len(methods)

while True:
    ret0, frame = camera.read()

    if inputCounter == 1:
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        retProcessed, processedImage = cv.threshold(grey,110,255,cv.THRESH_BINARY)

    else:
        processedImage = frame
        retProcessed = ret0
        
    cv.imshow("Cam 0", processedImage)
    vout.write(processedImage)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        inputCounter += 1

        if inputCounter > numMethods:
            break

# When everything is done, release the capture
cv.imwrite('./Cam0.jpg', processedImage)
camera.release()
cv.destroyAllWindows()
