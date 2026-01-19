import numpy as np


def IPT_adjust(IPT, XYZ):

    L_A = 0.2 * XYZ[..., 1]
    k = 1.0 / (5 * L_A + 1)
    F_L = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4) ** 2 * (5 * L_A) ** (1 / 3)

    C = np.sqrt(IPT[..., 1] ** 2 + IPT[..., 2] ** 2)

    adjustment = (F_L + 1) ** 0.2 * (
        (1.29 * C**2 - 0.27 * C + 0.42) / (C**2 - 0.31 * C + 0.42)
    )

    IPT[..., 1] = IPT[..., 1] * adjustment
    IPT[..., 2] = IPT[..., 2] * adjustment
    IPT[..., 0] = IPT[..., 0] ** 1.0

    return IPT
