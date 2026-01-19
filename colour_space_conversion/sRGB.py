import numpy as np


M_XYZ_to_sRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)

M_sRGB_to_XYZ = np.linalg.inv(M_XYZ_to_sRGB)


def XYZ_to_sRGB(XYZ):
    """
    Convert XYZ color space to sRGB color space.

    Parameters:
    XYZ : array-like
        Input array of XYZ color values. shape should be (..., 3)

    Returns:
    sRGB : array-like
        Output array of sRGB color values. shape will be the same as input XYZ
    """
    XYZ_reshape = XYZ.reshape((-1, 3))
    RGB = XYZ_reshape @ M_XYZ_to_sRGB.T
    
    # Apply gamma correction
    RGB = np.where(RGB <= 0.0031308, 12.92 * RGB, 1.055 * RGB ** (1 / 2.4) - 0.055)
    RGB = np.clip(RGB, 0, 1)
    
    sRGB = RGB.reshape(XYZ.shape)
    return sRGB


def sRGB_to_XYZ(sRGB):
    """
    Convert sRGB color space to XYZ color space.

    Parameters:
    sRGB : array-like
        Input array of sRGB color values. shape should be (..., 3)

    Returns:
    XYZ : array-like
        Output array of XYZ color values. shape will be the same as input sRGB
    """
    sRGB_reshape = sRGB.reshape((-1, 3))
    
    # Reverse gamma correction
    RGB_linear = np.where(sRGB_reshape <= 0.04045, 
                          sRGB_reshape / 12.92, 
                          ((sRGB_reshape + 0.055) / 1.055) ** 2.4)
    
    XYZ = RGB_linear @ M_sRGB_to_XYZ.T
    XYZ = XYZ.reshape(sRGB.shape)
    
    return XYZ
