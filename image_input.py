import cv2
import numpy as np
import colour


def read_tif_to_array(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img) / 65535.0
    return img


def read_img(file_name):
    img = read_tif_to_array(file_name)
    # Convert to XYZ
    img = colour.models.eotf_BT2100_PQ(img)
    img = colour.RGB_to_XYZ(img, colourspace="ITU-R BT.2020")
    return img
