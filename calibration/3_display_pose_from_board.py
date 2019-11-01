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

camera_matrix = np.array([[929.13251611,   0.,         479.17562521],
 [  0.,         931.26451127, 295.35871445],
 [  0.,           0.,           1.        ]])

distortion_coefficients = np.array([[ 1.35915086e-01, -2.23009579e+00, -1.37639118e-02, -2.29458613e-03,
   8.38818104e+00]])

# Get the width & height of camera stream
camera = cv2.VideoCapture(0)
ret, img = camera.read()
h, w = img.shape[:2]

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

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img