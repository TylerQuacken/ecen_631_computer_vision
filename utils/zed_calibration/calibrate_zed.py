'''
///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2018, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////
/*****************************************************************************************
 ** This sample demonstrates how to capture stereo images and calibration parameters    **
 ** from the ZED camera with OpenCV without using the ZED SDK.                          **
 *****************************************************************************************/
'''

import numpy as np
import os
import configparser
import sys
import cv2
import wget
import scipy.io as sio


def download_calibration_file(serial_number):
    if os.name == 'nt':
        hidden_path = os.getenv('APPDATA') + '\\Stereolabs\\settings\\'
    else:
        hidden_path = '/usr/local/zed/settings/'
    calibration_file = hidden_path + 'SN' + str(serial_number) + '.conf'

    if os.path.isfile(calibration_file) == False:
        url = 'http://calib.stereolabs.com/?SN='
        filename = wget.download(url=url + str(serial_number),
                                 out=calibration_file)

        if os.path.isfile(calibration_file) == False:
            print('Invalid Calibration File')
            return ""

    return calibration_file


def init_calibration(calibration_file, image_size):

    cameraMarix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array(
        [])

    config = configparser.ConfigParser()
    config.read(calibration_file)

    check_data = True
    resolution_str = ''
    if image_size.width == 2208:
        resolution_str = '2K'
    elif image_size.width == 1920:
        resolution_str = 'FHD'
    elif image_size.width == 1280:
        resolution_str = 'HD'
    elif image_size.width == 672:
        resolution_str = 'VGA'
    else:
        resolution_str = 'HD'
        check_data = False

    T_ = np.array([
        -float(config['STEREO']['Baseline'] if 'Baseline' in
               config['STEREO'] else 0),
        float(config['STEREO']['TY_' + resolution_str] if 'TY_' +
              resolution_str in config['STEREO'] else 0),
        float(config['STEREO']['TZ_' + resolution_str] if 'TZ_' +
              resolution_str in config['STEREO'] else 0)
    ])

    left_cam_cx = float(config['LEFT_CAM_' + resolution_str]['cx'] if 'cx' in
                        config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_cy = float(config['LEFT_CAM_' + resolution_str]['cy'] if 'cy' in
                        config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_fx = float(config['LEFT_CAM_' + resolution_str]['fx'] if 'fx' in
                        config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_fy = float(config['LEFT_CAM_' + resolution_str]['fy'] if 'fy' in
                        config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_k1 = float(config['LEFT_CAM_' + resolution_str]['k1'] if 'k1' in
                        config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_k2 = float(config['LEFT_CAM_' + resolution_str]['k2'] if 'k2' in
                        config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_p1 = float(config['LEFT_CAM_' + resolution_str]['p1'] if 'p1' in
                        config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_p2 = float(config['LEFT_CAM_' + resolution_str]['p2'] if 'p2' in
                        config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_p3 = float(config['LEFT_CAM_' + resolution_str]['p3'] if 'p3' in
                        config['LEFT_CAM_' + resolution_str] else 0)
    left_cam_k3 = float(config['LEFT_CAM_' + resolution_str]['k3'] if 'k3' in
                        config['LEFT_CAM_' + resolution_str] else 0)

    right_cam_cx = float(config['RIGHT_CAM_' + resolution_str]['cx'] if 'cx' in
                         config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_cy = float(config['RIGHT_CAM_' + resolution_str]['cy'] if 'cy' in
                         config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_fx = float(config['RIGHT_CAM_' + resolution_str]['fx'] if 'fx' in
                         config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_fy = float(config['RIGHT_CAM_' + resolution_str]['fy'] if 'fy' in
                         config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_k1 = float(config['RIGHT_CAM_' + resolution_str]['k1'] if 'k1' in
                         config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_k2 = float(config['RIGHT_CAM_' + resolution_str]['k2'] if 'k2' in
                         config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_p1 = float(config['RIGHT_CAM_' + resolution_str]['p1'] if 'p1' in
                         config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_p2 = float(config['RIGHT_CAM_' + resolution_str]['p2'] if 'p2' in
                         config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_p3 = float(config['RIGHT_CAM_' + resolution_str]['p3'] if 'p3' in
                         config['RIGHT_CAM_' + resolution_str] else 0)
    right_cam_k3 = float(config['RIGHT_CAM_' + resolution_str]['k3'] if 'k3' in
                         config['RIGHT_CAM_' + resolution_str] else 0)

    R_zed = np.array([
        float(config['STEREO']['RX_' + resolution_str] if 'RX_' +
              resolution_str in config['STEREO'] else 0),
        float(config['STEREO']['CV_' + resolution_str] if 'CV_' +
              resolution_str in config['STEREO'] else 0),
        float(config['STEREO']['RZ_' + resolution_str] if 'RZ_' +
              resolution_str in config['STEREO'] else 0)
    ])

    R, _ = cv2.Rodrigues(R_zed)
    cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                                  [0, left_cam_fy, left_cam_cy], [0, 0, 1]])

    cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                                   [0, right_cam_fy, right_cam_cy], [0, 0, 1]])

    distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1],
                                [left_cam_p2], [left_cam_k3]])

    distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2],
                                 [right_cam_p1], [right_cam_p2],
                                 [right_cam_k3]])

    T = np.array([[T_[0]], [T_[1]], [T_[2]]])
    R1 = R2 = P1 = P2 = np.array([])

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=cameraMatrix_left,
        cameraMatrix2=cameraMatrix_right,
        distCoeffs1=distCoeffs_left,
        distCoeffs2=distCoeffs_right,
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        imageSize=(image_size.width, image_size.height),
        newImageSize=(image_size.width, image_size.height))

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        cameraMatrix_left, distCoeffs_left, R1, P1,
        (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        cameraMatrix_right, distCoeffs_right, R2, P2,
        (image_size.width, image_size.height), cv2.CV_32FC1)

    data = {
        "cameraL": cameraMatrix_left,
        "cameraR": cameraMatrix_right,
        "distL": distCoeffs_left,
        "distR": distCoeffs_right,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "R": R,
        "T": T
    }

    sio.savemat("ZED_stereo_params.mat", data)
    print("Params Saved to File")

    cameraMatrix_left = P1
    cameraMatrix_right = P2

    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y


class Resolution:
    #HD settings
    width = 1280
    height = 720
    #VGA Settings
    # width = 672
    # height = 378
    #VGA 672x378 100 fps
    #HD720 1280x720 60 fps
    #1080  1920x1080 30 fps
    #2k 2208x1242 15fps


def main():

    # if len(sys.argv) == 1:
    #     print('Please provide ZED serial number')
    #     exit(1)

    # Open the ZED camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == 0:
        exit(-1)

    image_size = Resolution()
    # Christian Commented these 2 lines, I think the width and height are set in the public definition of Resolution,  shouldn't need to re-assign them, might want to include a method to set resolution based on an input at the start.
    #image_size.width = 1280
    #image_size.height = 720

    # Set the video resolution to HD720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)

    serial_number = 18741271
    calibration_file = download_calibration_file(serial_number)
    if calibration_file == "":
        exit(1)
    print("Calibration file found. Loading...")

    camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(
        calibration_file, image_size)

    while True:
        # Get a new frame from camera
        retval, frame = cap.read()

        if retval:
            # Extract left and right images from side-by-side
            left_right_image = np.split(frame, 2, axis=1)
            # Display images
            #cv2.imshow("left RAW", left_right_image[0])

            left_rect = cv2.remap(left_right_image[0],
                                  map_left_x,
                                  map_left_y,
                                  interpolation=cv2.INTER_LINEAR)
            right_rect = cv2.remap(left_right_image[1],
                                   map_right_x,
                                   map_right_y,
                                   interpolation=cv2.INTER_LINEAR)

            cv2.imshow("left RECT", left_rect)
            cv2.imshow("right RECT", right_rect)
        # Is this where the framerate is set? (google says that the value in waitkey is how long a frame is shown for so... basically).
        # VGA resolution max hz is 100 so 1/100=.01 s or 10 ms per frame (tried going down to 9 to see, it would show a line on fast movements.)
        if cv2.waitKey(10) >= 0:
            break

    exit(0)


if __name__ == "__main__":
    main()
