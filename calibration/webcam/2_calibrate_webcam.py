import cv2
import numpy as np

from cv2 import aruco
from pathlib import Path

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()

# Set path to the images
calibration_photo_path = root.joinpath("photos")

aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
aruco_params =  aruco.DetectorParameters_create()

# Params used to generate original board
board = aruco.GridBoard_create(5, 7, .04, .02, aruco_dict)

photo_list = []
calibration_photos = calibration_photo_path.glob('*.jpg')

# Loop and create the list of photos taken of the board by Tello
for id, file_name in enumerate(calibration_photos):
    photo = cv2.imread(str(root.joinpath(file_name)))
    photo_list.append(photo)
    h, w, c = photo.shape


counter, marker_corners_list, marker_id_list = [], [], []
first = True

for photo in photo_list:
    img_gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    marker_corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)

    if first == True:
        marker_corners_list = marker_corners
        marker_id_list = ids
        first = False
    else:
        # Append corners to corner list
        marker_corners_list = np.vstack((marker_corners_list, marker_corners))

        # Append marker ids to list
        marker_id_list = np.vstack((marker_id_list,ids))

    counter.append(len(ids))

counter = np.array(counter)

ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(marker_corners_list, marker_id_list, counter, board, img_gray.shape, None, None)

print("Camera matrix: \n", mtx)
print("Distortion coefficients: \n", dist)