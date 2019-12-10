import numpy
import cv2
from cv2 import aruco

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Create constants to be passed into OpenCV and Aruco methods
board = CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.04,
        markerLength=0.02,
        dictionary=ARUCO_DICT)

image = board.draw((1000, 1000))
cv2.imwrite("charuco_board.png", image)
