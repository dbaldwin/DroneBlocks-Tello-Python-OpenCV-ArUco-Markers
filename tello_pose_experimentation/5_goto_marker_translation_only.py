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

# Which marker to find
id_to_find  = 5
marker_size  = 17.5 # cm

# Load camera parameters
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

# Define the aruco dictionary
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters  = aruco.DetectorParameters_create()

# Capture the videocamera (this may also be a video or a picture)
# Get the video stream
#cap = cv2.VideoCapture(0) # webcam
cap = cv2.VideoCapture('udp://127.0.0.1:11111') # Tello video from stream
#cap = cv2.VideoCapture('../videos/tello.avi') # Tello video from file

# Set the camera size as the one it was calibrated with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

# We'll take a running sample every 32 camera frames
sample_count = 32
x_avg = []
y_avg = []
z_avg = []
min_x_distance, max_x_distance = (-10, 10) # Y in Tello's world
min_y_distance, max_y_distance = (-10, 10) # Z in Tello's world
min_z_distance, max_z_distance = (80, 100) # X in Tello's world

# Main loop
while True:

    # Read the camera frame
    ret, frame = cap.read()

    # Convert in gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                              cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    
    # Detect ID specified above
    if ids is not None and ids[0] == id_to_find:
        
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        # Unpack the output, get only the first
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

        # Draw the detected marker and put a reference frame over it
        aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

        # Print the marker position
        str_position = "Marker %d, x=%4.0f y=%4.0f z=%4.0f"%(ids[0], tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, str_position, (20, 20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        x_translation = tvec[0]
        y_translation = tvec[1]
        z_translation = tvec[2]

        x_avg.append(x_translation)
        y_avg.append(y_translation)
        z_avg.append(z_translation)

        if len(x_avg) == 31 and len(z_avg) == 31 and len(y_avg) == 31:
            x_translation = int(sum(x_avg)/len(x_avg))
            y_translation = int(sum(y_avg)/len(y_avg))
            z_translation = int(sum(z_avg)/len(z_avg))

            # Reset the arrays
            x_avg = []
            y_avg = []
            z_avg = []

            # We'll use this value to send to Tello
            x_offset = 0
            y_offset = 0
            z_offset = 0

            # X distance from marker. This is the Y axis (left/right) in Tello's world
            if x_translation < min_x_distance:
                x_offset = -20
            elif x_translation > max_x_distance:
                x_offset = 20

            # Y distance from marker. This is the Z axis (up/down) in Tello's world
            if y_translation < min_y_distance:
                y_offset = 20
            elif y_translation > max_y_distance:
                y_offset = -20

            # Z distance from marker. This is the X axis (forward/backward) in Tello's world
            if z_translation < min_z_distance:
                z_offset = -20
            elif z_translation > max_z_distance:
                z_offset = 20


            # The go SDK command in Tello operates as follows:
            # X represents forward(+) backward(-)
            # Y represents left right
            # Z represents up down

            if x_offset > 0 or y_offset > 0 or z_offset > 0:
                command = "go " + str(z_offset) + " " + str(x_offset) + " " + str(y_offset) + " 50"
                print(command)
                print(str_position)
                print("-------------------------------")
                send(command)



    # Display the frame
    cv2.imshow('frame', frame)

    # use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        break
    if key == ord(' '):
        send('takeoff')
