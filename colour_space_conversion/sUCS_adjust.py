import numpy as np


def sUCS_adjust(sUCS, XYZ):

    L_A = 0.2 * XYZ[..., 1]
    F_L = 0.171 * L_A ** (1 / 3) * (1 / (1 - 0.4934 * np.exp(-0.9934 * L_A)))

    C = (1 / 0.0252) * np.log(
        1 + 0.0447 * np.sqrt(sUCS[..., 1] ** 2 + sUCS[..., 2] ** 2)
    )

    adjustment = (F_L + 1) ** 0.2 * (
        (1.29 * C**2 - 0.27 * C + 0.42) / (C**2 - 0.31 * C + 0.42)
    )

    adjustment = (F_L + 1) ** 0.2 * (
        (1.29 * C**2 - 0.27 * C + 0.42) / (C**2 - 0.31 * C + 0.42)
    )

    sUCS[..., 1] = sUCS[..., 1] * adjustment
    sUCS[..., 2] = sUCS[..., 2] * adjustment
    sUCS[..., 0] = sUCS[..., 0] ** 1.0

    return sUCS
