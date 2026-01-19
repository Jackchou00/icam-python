import numpy as np


M_XYZ_to_LMS = np.array(
    [
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0000, 0.0000, 0.9184],
    ]
)

M_LMS_prime_to_Iab = np.array(
    [
        [2 / 3.05, 1 / 3.05, 0.05 / 3.05],
        [430, -470, 40],
        [49, 49, -98],
    ]
)

M_LMS_to_XYZ = np.linalg.inv(M_XYZ_to_LMS)
M_IPT_to_LMS_prime = np.linalg.inv(M_LMS_prime_to_Iab)


def XYZ_to_sUCS_Iab(XYZ):
    """
    Convert XYZ color space to sUCS color space.

    Parameters:
    XYZ : array-like
        Input array of XYZ color values. shape should be (..., 3)

    Returns:
    sUCS : array-like
        Output array of sUCS color values. shape will be the same as input XYZ
    """

    XYZ_reshape = XYZ.reshape((-1, 3))
    LMS = XYZ_reshape @ M_XYZ_to_LMS.T

    LMS_prime = np.sign(LMS) * np.abs(LMS) ** 0.43

    sUCS_Iab = LMS_prime @ M_LMS_prime_to_Iab.T
    sUCS_Iab = sUCS_Iab.reshape(XYZ.shape)

    return sUCS_Iab


def sUCS_Iab_to_ICh(sUCS_Iab):

    C = (1 / 0.0252) * np.log(
        1 + 0.0447 * np.sqrt(sUCS_Iab[..., 1] ** 2 + sUCS_Iab[..., 2] ** 2)
    )

    h = np.arctan2(sUCS_Iab[..., 2], sUCS_Iab[..., 1])
    h_deg = np.degrees(h)
    h_deg[h_deg < 0] += 360
    ICh = np.zeros_like(sUCS_Iab)
    ICh[..., 0] = sUCS_Iab[..., 0]
    ICh[..., 1] = C
    ICh[..., 2] = h_deg
    return ICh


def sUCS_Iab_to_XYZ(sUCS_Iab):
    """
    Convert sUCS color space to XYZ color space.

    Parameters:
    sUCS_Iab : array-like
        Input array of sUCS color values. shape should be (..., 3)

    Returns:
    XYZ : array-like
        Output array of XYZ color values. shape will be the same as input sUCS_Iab
    """
    sUCS_reshape = sUCS_Iab.reshape((-1, 3))
    LMS_prime = sUCS_reshape @ M_IPT_to_LMS_prime.T

    LMS = np.sign(LMS_prime) * np.abs(LMS_prime) ** (1 / 0.43)
    XYZ = LMS @ M_LMS_to_XYZ.T
    XYZ = XYZ.reshape(sUCS_Iab.shape)

    return XYZ
