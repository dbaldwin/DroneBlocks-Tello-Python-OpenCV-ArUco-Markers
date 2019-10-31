import cv2
from cv2 import aruco
import numpy as np

aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
aruco_params =  aruco.DetectorParameters_create()

camera_matrix = np.array([[927.32436751,    0.,         491.7253811],
  [  0.,         927.60209717, 302.39067493],
  [  0.,           0.,           1.        ]])

distortion_coefficients = np.array( [[ 0.02255412, -0.2682463,  -0.00997881,  0.0011136,    0.8426477]])

# Get the width & height of camera stream
camera = cv2.VideoCapture(0)
ret, img = camera.read()
h, w = img.shape[:2]

while True:
    ret, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    marker_corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        img_aruco = aruco.drawDetectedMarkers(img, marker_corners, ids, (0, 255, 0))
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(marker_corners, 4, camera_matrix, distortion_coefficients)

        # In case there are multiple markers
        for i in range(ids.size):
            img_aruco = aruco.drawAxis(img_aruco, camera_matrix, distortion_coefficients, rvec[i], tvec[i], 2)

        print("rvec: ", rvec)
        print("tvec: ", tvec)

    else:
        img_aruco = img

    cv2.imshow("Aruco Marker Detection", img_aruco)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()