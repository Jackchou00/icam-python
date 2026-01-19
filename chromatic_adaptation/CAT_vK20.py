import numpy as np


M_CAT16 = np.array(
    [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ]
)

M_CAT16_inv = np.linalg.inv(M_CAT16)


def img_vK20_to_D65(XYZ, XYZ_w, surround="average"):
    # XYZ input: shape (h, w, 3)
    # XYZ_w: shape (h, w, 3)
    # D: shape (h, w)
    D_p = 0.0
    D_n = 0.7
    D_r = 0.3

    XYZ_p = np.array([0.95045593, 1.00000000, 1.08905775])
    XYZ_D65 = np.array([0.95047, 1.00000, 1.08883])
    XYZ_r = np.array([0.97941176, 1.00000000, 1.73235294])

    LMS_r = XYZ_r @ M_CAT16.T
    LMS_D65 = XYZ_D65 @ M_CAT16.T

    XYZ_reshape = XYZ.reshape((-1, 3))
    LMS = XYZ_reshape @ M_CAT16.T
    LMS = LMS.reshape(XYZ.shape)

    XYZ_w_reshape = XYZ_w.reshape((-1, 3))
    LMS_w = XYZ_w_reshape @ M_CAT16.T
    LMS_w = LMS_w / np.max(LMS_w)
    LMS_w = LMS_w.reshape(XYZ_w.shape)

    LMS_c = np.zeros_like(LMS)
    for i in range(3):
        k = (D_r * LMS_r[i] + D_n * LMS_D65[i]) / (D_r * LMS_r[i] + D_n * LMS_w[..., i])
        LMS_c[..., i] = k * LMS[..., i]

    # reshape LMS to (3, h*w)
    LMS_c_reshape = LMS_c.reshape((-1, 3))
    XYZ_c = LMS_c_reshape @ M_CAT16_inv.T
    XYZ_c = XYZ_c.reshape(XYZ.shape)

    return XYZ_c
