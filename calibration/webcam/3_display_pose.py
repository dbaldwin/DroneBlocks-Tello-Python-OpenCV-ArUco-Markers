import cv2
from cv2 import aruco
import numpy as np
from pathlib import Path

root = Path(__file__).parent.absolute()

calibration_photo_path = root.joinpath("photos")

marker_length = .04
marker_spacing = .02
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
aruco_params =  aruco.DetectorParameters_create()

board = aruco.GridBoard_create(5, 7, marker_length, marker_spacing, aruco_dict)

camera_matrix = np.array([[937.16789186,   0.,         611.68079247],
 [  0.,         935.47927635, 397.88875823],
 [  0.,         0.,           1.        ]])

distortion_coefficients = np.array([[-0.04809431,  0.09007492,  0.01433798, -0.00804455, -0.20395533]])

# Get the video stream
camera = cv2.VideoCapture(0) # webcam
#camera = cv2.VideoCapture('udp://127.0.0.1:11111') # Tello video from stream
#camera = cv2.VideoCapture('../videos/tello.avi') # Tello video from file

while True:
    ret, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    marker_corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        ret, rvec, tvec = aruco.estimatePoseBoard(marker_corners, ids, board, camera_matrix, distortion_coefficients, None, None)
        img_aruco = aruco.drawDetectedMarkers(img, marker_corners, ids, (0, 255, 0))

        if ret != 0:
            img_aruco = aruco.drawAxis(img_aruco, camera_matrix, distortion_coefficients, rvec, tvec, marker_length)
    else:
        img_aruco = img

    cv2.imshow("Aruco Marker Detection", img_aruco)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()