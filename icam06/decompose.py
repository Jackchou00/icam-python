import numpy as np
import cv2


def bilateral_filter(img):
    img[img < 1e-4] = 1e-4
    logimg = np.log10(img)

    sigmaColor = 0.35
    if min(img.shape) < 1024:
        z = 2
    else:
        z = 4
    _, xDim, _ = img.shape
    sigmaSpace = 2 * xDim / z / 100
    base_layer = cv2.bilateralFilter(
        logimg, d=-1, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace
    )

    detail_layer = logimg - base_layer
    detail_layer[detail_layer > 12] = 0

    base_layer = np.power(10, base_layer)
    detail_layer = np.power(10, detail_layer)

    return base_layer, detail_layer
