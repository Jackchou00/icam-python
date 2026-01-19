import numpy as np


M_XYZ_to_LMS = np.array(
    [
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0000, 0.0000, 0.9184],
    ]
)

M_LMS_prime_to_IPT = np.array(
    [
        [0.4000, 0.4000, 0.2000],
        [4.4550, -4.8510, 0.3960],
        [0.8056, 0.3572, -1.1628],
    ]
)

M_LMS_to_XYZ = np.linalg.inv(M_XYZ_to_LMS)
M_IPT_to_LMS_prime = np.linalg.inv(M_LMS_prime_to_IPT)


def XYZ_to_IPT(XYZ):
    """
    Convert XYZ color space to IPT color space.

    Parameters:
    XYZ : array-like
        Input array of XYZ color values. shape should be (..., 3)

    Returns:
    IPT : array-like
        Output array of IPT color values. shape will be the same as input XYZ
    """

    XYZ_reshape = XYZ.reshape((-1, 3))
    LMS = XYZ_reshape @ M_XYZ_to_LMS.T

    LMS_prime = np.sign(LMS) * np.abs(LMS) ** 0.43

    IPT = LMS_prime @ M_LMS_prime_to_IPT.T
    IPT = IPT.reshape(XYZ.shape)

    return IPT


def IPT_to_XYZ(IPT):
    """
    Convert IPT color space to XYZ color space.

    Parameters:
    IPT : array-like
        Input array of IPT color values. shape should be (..., 3)

    Returns:
    XYZ : array-like
        Output array of XYZ color values. shape will be the same as input IPT
    """

    IPT_reshape = IPT.reshape((-1, 3))
    LMS_prime = IPT_reshape @ M_IPT_to_LMS_prime.T

    LMS = np.sign(LMS_prime) * np.abs(LMS_prime) ** (1 / 0.43)
    XYZ = LMS @ M_LMS_to_XYZ.T
    XYZ = XYZ.reshape(IPT.shape)

    return XYZ
