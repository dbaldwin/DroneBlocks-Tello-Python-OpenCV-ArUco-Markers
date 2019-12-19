import cv2
from cv2 import aruco
import numpy as np
import math
import os
import pickle
import time
from lib.tello import Tello

# Construct a Tello instance so we can communicate with it over UDP
tello = Tello()

# Send the command string to wake Tello up
tello.send("command")

# Delay
time.sleep(1)

# Initialize the video stream which will start sending to port 11111
tello.send("streamon")

marker_length = 17.5 # 175mm/17.5cm
aruco_params = aruco.DetectorParameters_create()
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

# Check for camera calibration data
if not os.path.exists('./tello_calibration.pckl'):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:
    f = open('tello_calibration.pckl', 'rb')
    (camera_matrix, distortion_coefficients, _, _) = pickle.load(f)
    f.close()
    if camera_matrix is None or distortion_coefficients is None:
        print("Calibration issue. Remove ./tello_calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
        exit()

# Get Tello video stream
cam = cv2.VideoCapture('udp://127.0.0.1:11111')
#cam = cv2.VideoCapture('./utils/20191219-104330.avi')
    
while True:
    ret, img = cam.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    marker_corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        img_aruco = aruco.drawDetectedMarkers(img, marker_corners, ids, (0, 255, 0))

        # tvec is the center of the marker in the camera's world
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(marker_corners, marker_length, camera_matrix, distortion_coefficients)

        # In case there are multiple markers
        for i in range(ids.size):
            img_aruco = aruco.drawAxis(img_aruco, camera_matrix, distortion_coefficients, rvec[i], tvec[i], marker_length)

            # Plot a point at the center of the image
            cv2.circle(img_aruco, (480, 360), 3, (255, 255, 0), -1)

            #cv2.rectangle(img_aruco, (0, 620), (200, 720), (0, 0, 0), -1)



        # if rvec.size == 3:
        #     imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, distortion_coefficients)

        #     p1 = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
                
        #     if p1 is not None:
        #         cv2.circle(img_aruco, p1, 5, (0, 0, 255), -1)

        #     p2 = (int(imgpts[1][0][0]), int(imgpts[1][0][1]))

        #     if p2 is not None:
        #         cv2.circle(img_aruco, p2, 5, (0, 255, 0), -1)

        #     p3 = (int(imgpts[2][0][0]), int(imgpts[2][0][1]))
            
        #     if p3 is not None:
        #         cv2.circle(img_aruco, p3, 5, (255, 0, 0), -1)

        #     # Plot a point at the center of the image
        #     cv2.circle(img_aruco, (480, 360), 5, (0, 255, 0), -1)

        #     tvec_x = tvec[0][0][0]
        #     tvec_y = tvec[0][0][1]
        #     tvec_z = tvec[0][0][2]

        #     # Distance from camera is the magnitude of tvec
        #     distance = math.sqrt(tvec_x*tvec_x + tvec_y*tvec_y + tvec_z*tvec_z)
        #     print(distance)

        #     # Let's focus on keeping the marker centered on the x axis (roll left/right)
        #     # This means we'll consider y and z constant for this demonstration

        #     # Calculate angle of vectors
        #     array = np.array([tvec_x, tvec_y, tvec_z])
        #     array_mag = np.linalg.norm(array)

        #     # Vector to center of screen
        #     array2 = np.array([0, 0, tvec_z])
        #     array2_mag = np.linalg.norm(array2)

        #     dot = np.dot(array, array2)

        #     # Solve for angle
        #     cos = np.arccos(dot/(array_mag*array2_mag))

        #     degrees = np.degrees(cos)

            #print(degrees)




        #for p in marker_corners:

            # Draw line from center of camera to first corner
            #cv2.arrowedLine(img_aruco, (480, 360), (int(p[0][0][0]), int(p[0][0][1])), (0, 0, 255), 3)


        #print(x, ":", y)

        #img_aruco = cv2.line(img_aruco, (0, 0), (int(x), int(y)), (0,0,255), 3)

        #print("rvec: ", rvec)
        #print("tvec: ", tvec)

    else:
        img_aruco = img

    cv2.imshow("Tello", img_aruco)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Press space bar to take photo
    if key == ord(' '):
        file_path = os.getcwd()
        file_name = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(file_path + "/" + file_name + ".jpg", img_aruco)

cv2.destroyAllWindows()