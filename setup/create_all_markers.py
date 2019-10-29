# This script was used to create markers from 0-1023 in the /markers folder
# You can use this script to create your own using a different dictionary
# Or perhaps you only want to create a subset of markers

import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl

starting_marker_id = 0
ending_marker_id = 1024
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

for i in range(starting_marker_id, ending_marker_id):
    fig = plt.figure()
    img = aruco.drawMarker(aruco_dict, i, 100)
    plt.axis("off")
    plt.imshow(img, cmap = mpl.cm.gray)
    plt.savefig("../markers/" + str(i) + ".png")