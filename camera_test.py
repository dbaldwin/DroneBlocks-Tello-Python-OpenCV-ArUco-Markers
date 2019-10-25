import cv2
import numpy as np

# Must enable the Rpi camera first in raspi-config and reboot

# For RPi users be sure to execute this command to enable camera
# sudo modprobe bcm2835-v4l2
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
