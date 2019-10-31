# File to create ArUco grid for camera calibration
# Running the script will create board.png
# Print board.png and begin the calibration process
# Parameters taken from https://docs.opencv.org/master/db/da9/tutorial_aruco_board_detection.html
# This will create a board with markers 0 - 34

import cv2
from cv2 import aruco

marker_length = 4
marker_spacing = 1
grid_cols = 5
grid_rows = 7
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
board = aruco.GridBoard_create(grid_cols, grid_rows, marker_length, marker_spacing, aruco_dict)
img = board.draw((864,1080))
cv2.imshow("aruco", img)
cv2.imwrite("board.png", img)
cv2.waitKey()