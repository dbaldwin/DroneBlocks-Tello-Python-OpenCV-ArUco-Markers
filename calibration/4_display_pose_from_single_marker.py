import cv2
from cv2 import aruco
import numpy as np

marker_length = .04
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
aruco_params =  aruco.DetectorParameters_create()

camera_matrix = np.array([[929.13251611,   0.,         479.17562521],
 [  0.,         931.26451127, 295.35871445],
 [  0.,           0.,           1.        ]])

distortion_coefficients = np.array([[ 1.35915086e-01, -2.23009579e+00, -1.37639118e-02, -2.29458613e-03,
   8.38818104e+00]])

# Get the video stream
#camera = cv2.VideoCapture(0) # webcam
camera = cv2.VideoCapture('udp://127.0.0.1:11111') # Tello

while True:
    ret, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    marker_corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        img_aruco = aruco.drawDetectedMarkers(img, marker_corners, ids, (0, 255, 0))
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(marker_corners, marker_length, camera_matrix, distortion_coefficients)

        # In case there are multiple markers
        for i in range(ids.size):
            img_aruco = aruco.drawAxis(img_aruco, camera_matrix, distortion_coefficients, rvec[i], tvec[i], marker_length)

        #img_aruco = cv2.line(img_aruco, (0, 0), (int(x*10), int(y*10)), (0,0,255), 3)

        #print("rvec: ", rvec)
        print("tvec: ", tvec)

    else:
        img_aruco = img

    cv2.imshow("Aruco Marker Detection", img_aruco)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()