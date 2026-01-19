import numpy as np
from colour.models import XYZ_to_RGB


def XYZ_to_P3_RGB(XYZ):
    XYZ = np.clip(XYZ / 100, 0, 1)
    P3_RGB = XYZ_to_RGB(XYZ, colourspace="Display P3", apply_cctf_encoding=True)
    RGB = np.clip(P3_RGB, 0, 1)
    return RGB
