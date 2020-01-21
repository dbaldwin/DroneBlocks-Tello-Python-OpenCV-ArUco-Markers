# Use this script to take photos of your calibration board
import cv2
from cv2 import aruco
import numpy as np
import math
import os
import time

camera = cv2.VideoCapture(0) # webcam
file_path = os.getcwd()

while True:
    ret, img = camera.read()

    cv2.imshow("Aruco Marker Detection", img)

    # Press q key to abort script
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        print("taking photo")
        filename = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(file_path + "/photos/" + filename + ".jpg", img)
    elif key & 0XFF == ord('q'):
        break

cv2.destroyAllWindows()