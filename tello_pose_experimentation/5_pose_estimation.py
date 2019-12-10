"""
This demo calculates multiple things for different scenarios.

Here are the defined reference frames:

TAG:
                A y
                |
                |
                |tag center
                O---------> x

CAMERA:


                X--------> x
                | frame center
                |
                |
                V y

F1: Flipped (180 deg) tag frame around x axis
F2: Flipped (180 deg) camera frame around x axis

The attitude of a generic frame 2 respect to a frame 1 can obtained by calculating euler(R_21.T)

We are going to obtain the following quantities:
    > from aruco library we obtain tvec and Rct, position of the tag in camera frame and attitude of the tag
    > position of the Camera in Tag axis: -R_ct.T*tvec
    > Transformation of the camera, respect to f1 (the tag flipped frame): R_cf1 = R_ct*R_tf1 = R_cf*R_f
    > Transformation of the tag, respect to f2 (the camera flipped frame): R_tf2 = Rtc*R_cf2 = R_tc*R_f
    > R_tf1 = R_cf2 an symmetric = R_f


"""

import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math, os, pickle
import socket
import threading

# IP and port of Tello
tello_address = ('192.168.10.1', 8889)
# Create a UDP connection that we'll send the command to
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Let's be explicit and bind to a local port on our machine where Tello can send messages
sock.bind(('', 9000))

def send(message):
  try:
    sock.sendto(message.encode(), tello_address)
    print("Sending message: " + message)
  except Exception as e:
    print("Error sending: " + str(e))

# Receive the message from Tello
def receive():
  # Continuously loop and listen for incoming messages
  while True:
    # Try to receive the message otherwise print the exception
    try:
      response, _ = sock.recvfrom(128)
      print("Received message: " + response.decode(encoding='utf-8'))
    except Exception as e:
      # If there's an error close the socket and break out of the loop
      sock.close()
      print("Error receiving: " + str(e))
      break

receiveThread = threading.Thread(target=receive)
receiveThread.daemon = True
receiveThread.start()

send("command")
time.sleep(1)
send("streamon")



#--- Define Tag
id_to_find  = 0
marker_size  = 10 #- [cm]

#------------------------------------------------------------------------------
#------- ROTATIONS https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#------------------------------------------------------------------------------
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


#--- Get the camera calibration path
# calib_path  = ""
# camera_matrix   = np.loadtxt(calib_path+'cameraMatrix_webcam.txt', delimiter=',')
# camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_webcam.txt', delimiter=',')

# Check for camera calibration data
if not os.path.exists('./tello_calibration.pckl'):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:
    f = open('tello_calibration.pckl', 'rb')
    (camera_matrix, camera_distortion, _, _) = pickle.load(f)
    f.close()
    if camera_matrix is None or camera_distortion is None:
        print("Calibration issue. Remove ./tello_calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
        exit()

#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

#--- Define the aruco dictionary
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
parameters  = aruco.DetectorParameters_create()


#--- Capture the videocamera (this may also be a video or a picture)
# Get the video stream
#cap = cv2.VideoCapture(0) # webcam
cap = cv2.VideoCapture('udp://127.0.0.1:11111') # Tello video from stream
#cap = cv2.VideoCapture('../videos/tello.avi') # Tello video from file

#-- Set the camera size as the one it was calibrated with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

while True:

    #-- Read the camera frame
    ret, frame = cap.read()

    #-- Convert in gray scale
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red

    #-- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                              cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    
    if ids is not None and ids[0] == id_to_find:
        
        #-- ret = [rvec, tvec, ?]
        #-- array of rotation and position of each marker in camera frame
        #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
        #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        #-- Unpack the output, get only the first
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

        #-- Draw the detected marker and put a reference frame over it
        aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

        #-- Print the tag position in camera frame
        str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, str_position, (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        min_distance, max_distance = (60, 80)

        if tvec[2] < min_distance:
            send("back 20")
            print("flying backward: current_distance is: " + str(tvec[2]))
        elif tvec[2] > max_distance:
            send("forward 20")
            print("flying forward: current_distance is: " + str(tvec[2]))

        # #-- Obtain the rotation matrix tag->camera
        # R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
        # R_tc    = R_ct.T

        # #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
        # roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)

        # #-- Print the marker's attitude respect to camera frame
        # str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),
        #                     math.degrees(yaw_marker))
        # cv2.putText(frame, str_attitude, (0, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


        # #-- Now get Position and attitude f the camera respect to the marker
        # pos_camera = -R_tc*np.matrix(tvec).T

        # str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f"%(pos_camera[0], pos_camera[1], pos_camera[2])
        # cv2.putText(frame, str_position, (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # #-- Get the attitude of the camera respect to the frame
        # roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip*R_tc)
        # str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_camera),math.degrees(pitch_camera),
        #                     math.degrees(yaw_camera))
        # cv2.putText(frame, str_attitude, (0, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)



    #--- Display the frame
    cv2.imshow('frame', frame)

    #--- use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        break
    if key == ord(' '):
        send('takeoff')
