import numpy as np


M_H_D65 = np.array(
    [
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0000, 0.0000, 0.9184],
    ]
)

M_IPT = np.array(
    [
        [0.4000, 0.4000, 0.2000],
        [4.4550, -4.8510, 0.3960],
        [0.8056, 0.3572, -1.1628],
    ]
)

M_H_D65_inv = np.linalg.inv(M_H_D65)
M_IPT_inv = np.linalg.inv(M_IPT)


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
    LMS = XYZ_reshape @ M_H_D65.T

    LMS_prime = np.sign(LMS) * np.abs(LMS) ** 0.43

    IPT = LMS_prime @ M_IPT.T
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
    LMS_prime = IPT_reshape @ M_IPT_inv.T

    LMS = np.sign(LMS_prime) * np.abs(LMS_prime) ** (1 / 0.43)
    XYZ = LMS @ M_H_D65_inv.T
    XYZ = XYZ.reshape(IPT.shape)

    return XYZ


def get_gamma(surround):
    surround_dict = {"average": 1.0, "dim": 1.25, "dark": 1.5}
    # default to 1.0 if surround is not defined
    return surround_dict.get(surround, 1.0)


def IPT_adjust(XYZ, surround="average"):
    # XYZ: shape (h, w, 3)

    # convert XYZ to IPT space
    IPT = XYZ_to_IPT(XYZ)

    # calculate Chroma and F_L
    C = np.sqrt(IPT[..., 1] ** 2 + IPT[..., 2] ** 2)

    L_A = 0.2 * XYZ[..., 1]
    k = 1.0 / (5 * L_A + 1)
    F_L = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4) ** 2 * (5 * L_A) ** (1 / 3)

    # P and T adjustment
    adjustment = (F_L + 1) ** 0.2 * (
        (1.29 * C**2 - 0.27 * C + 0.42) / (C**2 - 0.31 * C + 0.42)
    )
    IPT[..., 1] = IPT[..., 1] * adjustment
    IPT[..., 2] = IPT[..., 2] * adjustment

    # I adjustment
    IPT[..., 0] = IPT[..., 0] ** get_gamma(surround)

    # convert IPT back to XYZ space
    XYZ = IPT_to_XYZ(IPT)
    return XYZ
