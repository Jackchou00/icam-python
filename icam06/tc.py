import numpy as np


M_HPE = np.array(
    [
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.0, 0.0, 1.0],
    ]
)
M_HPE_inv = np.linalg.inv(M_HPE)


def tone_compression(XYZ, XYZ_w, p=0.75):
    # XYZ input: shape (h, w, 3)
    # XYZ_w: shape (h, w, 3), local adapted white image
    # p: user-controllable variable, larger value means higher contrast.
    # p range: 0.6-0.85, default 0.75
    # XYZ_TC: shape (h, w, 3), tone compressed tristimulus values

    # convert XYZ to RGB_prime
    XYZ_reshape = XYZ.reshape((-1, 3))
    RGB_prime = XYZ_reshape @ M_HPE.T
    RGB_prime = RGB_prime.reshape(XYZ.shape)

    L_A = 0.2 * XYZ_w[..., 1]
    k = 1.0 / (5 * L_A + 1)
    F_L = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4) ** 2 * (5 * L_A) ** (1 / 3)
    Y_w = XYZ_w[..., 1]

    # Cone post adaptation response
    # RGB_prime_a = 400 * (F_L * RGB_prime / Y_w) ** p / (27.13 + (F_L * RGB_prime / Y_w) ** p) + 0.1
    RGB_prime_a = np.zeros_like(RGB_prime)
    for i in range(3):
        RGB_prime_a[..., i] = (
            400
            * (F_L * RGB_prime[..., i] / Y_w) ** p
            / (27.13 + (F_L * RGB_prime[..., i] / Y_w) ** p)
            + 0.1
        )

    # Rod response
    S = XYZ[..., 1]
    Sw = np.max(XYZ_w[..., 1])

    # There are some numerical errors in the original code (MATLAB)
    # Here referred to the original paper instead of the MATLAB code
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

    # RGB_TC = RGB_prime_a + A_S
    RGB_TC = np.zeros_like(RGB_prime)
    for i in range(3):
        RGB_TC[..., i] = RGB_prime_a[..., i] + A_S

    # convert RGB_TC back to XYZ space
    RGB_TC_reshape = RGB_TC.reshape((-1, 3))
    XYZ_TC = RGB_TC_reshape @ M_HPE_inv.T
    XYZ_TC = XYZ_TC.reshape(XYZ.shape)
    return XYZ_TC
