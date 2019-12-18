# 1. Power up Tello
# 2. Connect to Tello network
# 3. Run this script with Python

import cv2
import os
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

# Path for storing file
file_path = os.getcwd()

# Start the video capture
cap = cv2.VideoCapture('udp://127.0.0.1:11111')

# Loop and grab image frames
while True:

    # Grab frame
    ret, frame = cap.read()

    # Show frame
    cv2.imshow('Tello', frame)

    # Press q key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        tello.close()
        break

    # Press space bar to take photo
    if key == ord(' '):
        file_name = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(file_path + "/" + file_name + ".jpg", frame)