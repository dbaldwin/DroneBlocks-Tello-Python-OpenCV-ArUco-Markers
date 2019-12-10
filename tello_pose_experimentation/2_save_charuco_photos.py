import cv2
import os
import time


file_path = os.getcwd()
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('udp://127.0.0.1:11111') # Tello video from stream
#cap = cv2.VideoCapture('../videos/tello.avi') # Tello video from file

while True:
    ret, frame = cap.read()

    cv2.imshow('camera', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        file_name = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(file_path + "/" + file_name + ".jpg", frame)