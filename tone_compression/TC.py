import numpy as np


M_HPE = np.array(
    [
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.0, 0.0, 1.0],
    ]
)


def img_TC(XYZ, white, p):
    XYZ_reshape = XYZ.reshape((-1, 3))
    RGB_dash = XYZ_reshape @ M_HPE.T
    RGB_dash = RGB_dash.reshape(XYZ.shape)

    L_A = 0.2 * white[..., 1]
    k = 1.0 / (5 * L_A + 1)
    F_L = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4) ** 2 * (5 * L_A) ** (1 / 3)
    Y_w = white[..., 1]

    # RGB_dash_a = 400 * (F_L * RGB_dash / Y_w) ** p / (27.13 + (F_L * RGB_dash / Y_w) ** p) + 0.1
    RGB_dash_a = np.zeros_like(RGB_dash)
    for i in range(3):
        RGB_dash_a[..., i] = (
            400
            * (F_L * RGB_dash[..., i] / Y_w) ** p
            / (27.13 + (F_L * RGB_dash[..., i] / Y_w) ** p)
            + 0.1
        )

    S = XYZ[..., 1]
    Sw = np.max(white[..., 1])

    L_AS = 2.26 * L_A
    j = 0.00001 / (5 * L_AS / 2.26 + 0.00001)
    B_S = 0.5 / (1 + 0.3 * (5 * L_AS / 2.26) * (S / Sw) ** 0.3) + 0.5 / (
        1 + 5 * (5 * L_AS / 2.26)
    )
    F_LS = 3800 * j**2 * (5 * L_AS / 2.26) + 0.2 * (1 - j**2) ** 4 * (
        5 * L_AS / 2.26
    ) ** (1 / 6)
    A_S = (
        3.05 * B_S * (400 * (F_LS * S / Sw) ** p / (27.13 + (F_LS * S / Sw) ** p)) + 0.3
    )

    # RGB_TC = RGB_dash_a + A_S
    RGB_TC = np.zeros_like(RGB_dash)
    for i in range(3):
        RGB_TC[..., i] = RGB_dash_a[..., i] + A_S

    RGB_TC_reshape = RGB_TC.reshape((-1, 3))
    XYZ_TC = RGB_TC_reshape @ np.linalg.inv(M_HPE).T
    XYZ_TC = XYZ_TC.reshape(XYZ.shape)
    return XYZ_TC
