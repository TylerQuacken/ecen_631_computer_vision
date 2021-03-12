import cv2
import numpy as np

loadpath = "../images/baseball_trajectory/BaseBall_Pitch_"
filenameL = loadpath + "L.avi"
filenameR = loadpath + "R.avi"

savePathL = "../images/baseball_trajectory/L/"
savePathR = "../images/baseball_trajectory/R/"
saveName = "{:02d}.png"


def save_frames(filename, savepath, savename):
    cap = cv2.VideoCapture(filename)

    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        cv2.imshow('frame', frame)

        cv2.imwrite(savepath + saveName.format(i), frame)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# save_frames(filenameL, savePathL, saveName)
save_frames(filenameR, savePathR, saveName)
