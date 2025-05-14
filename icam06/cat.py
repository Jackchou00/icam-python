import numpy as np


M_CAT02 = np.array(
    [
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.0030, 0.0136, 0.9834],
    ]
)
M_CAT02_inv = np.linalg.inv(M_CAT02)
# reference white point: D65
XYZ_wr = np.array([95.05, 100.0, 108.88])
LMS_wr = M_CAT02 @ XYZ_wr


def get_F(surround):
    surround_dict = {"average": 1.0, "dim": 0.9, "dark": 0.8}
    # default to 1.0 if surround is not defined
    return surround_dict.get(surround, 1.0)


def calc_D(XYZ_w, surround):
    F = get_F(surround)  # F is the surround factor, F=1.0 for average
    L_A = 0.2 * XYZ_w[..., 1]  # 20% of the adaptation white
    D = 0.3 * F * (1 - (1 / 3.6) * np.exp(-(L_A - 42) / 92))
    # 0.3 is applied to reduce the color de-saturation for HDR image rendering.
    # should be (L_A + 42) in correct CAT02
    return D


def CAT02_to_D65(XYZ, XYZ_w, surround="average"):
    # XYZ input: shape (h, w, 3)
    # XYZ_w: shape (h, w, 3), adaptation white point
    # XYZ_c: shape (h, w, 3), adapted tristimulus values

    # convert XYZ to LMS
    XYZ_reshape = XYZ.reshape((-1, 3))
    LMS_reshape = XYZ_reshape @ M_CAT02.T
    LMS = LMS_reshape.reshape(XYZ.shape)

    # convert XYZ_w to LMS_w
    XYZ_w_reshape = XYZ_w.reshape((-1, 3))
    LMS_w_reshape = XYZ_w_reshape @ M_CAT02.T
    LMS_w = LMS_w_reshape.reshape(XYZ_w.shape)

    LMS_c = np.zeros_like(LMS)
    # D: shape (h, w)
    D = calc_D(XYZ_w, surround)
    for i in range(3):
        # LMS_wr is normalized to 100, but LMS_w, LMS are absolute values
        # if D isn't scaled by factor 0.3, LMS_c will be normalized instead of absolute
        # if LMS_c is normalized, there will be numerical error in next step
        LMS_c[..., i] = (D * LMS_wr[i] / LMS_w[..., i] + 1 - D) * LMS[..., i]

    # convert LMS_c to XYZ_c
    LMS_c_reshape = LMS_c.reshape((-1, 3))
    XYZ_c = LMS_c_reshape @ M_CAT02_inv.T
    XYZ_c = XYZ_c.reshape(XYZ.shape)

    return XYZ_c


# here is a bug fixed version of CAT02_to_D65


def calc_D_fixed(XYZ_w, surround):
    F = get_F(surround)  # F is the surround factor, F=1.0 for average
    L_A = 0.2 * XYZ_w[..., 1]  # 20% of the adaptation white
    D = F * (1 - (1 / 3.6) * np.exp(-(L_A + 42) / 92))
    # 0.3 is not applied.
    # should be (L_A + 42) in correct CAT02
    return D


def CAT02_to_D65_fixed(XYZ, XYZ_w, surround="average"):
    # XYZ input: shape (h, w, 3)
    # XYZ_w: shape (h, w, 3), adaptation white point
    # XYZ_c: shape (h, w, 3), adapted tristimulus values

    # convert XYZ to LMS
    XYZ_reshape = XYZ.reshape((-1, 3))
    LMS_reshape = XYZ_reshape @ M_CAT02.T
    LMS = LMS_reshape.reshape(XYZ.shape)

    # convert XYZ_w to LMS_w
    XYZ_w_reshape = XYZ_w.reshape((-1, 3))
    LMS_w_reshape = XYZ_w_reshape @ M_CAT02.T
    LMS_w = LMS_w_reshape.reshape(XYZ_w.shape)

    LMS_c = np.zeros_like(LMS)
    # D: shape (h, w), but not use 0.3 to scale.
    D = calc_D_fixed(XYZ_w, surround)
    # add a factor to change the XYZ_wr as absolute value instead of normalized to 100
    Yw_Ywr = XYZ_w[..., 1] / XYZ_wr[1]
    for i in range(3):
        # LMS_wr is normalized to 100, but LMS_w, LMS are absolute values
        # if D isn't scaled by factor 0.3, LMS_c will be normalized instead of absolute
        # if LMS_c is normalized, there will be numerical error in next step
        LMS_c[..., i] = (D * Yw_Ywr * LMS_wr[i] / LMS_w[..., i] + 1 - D) * LMS[..., i]

    # convert LMS_c to XYZ_c
    LMS_c_reshape = LMS_c.reshape((-1, 3))
    XYZ_c = LMS_c_reshape @ M_CAT02_inv.T
    XYZ_c = XYZ_c.reshape(XYZ.shape)

    return XYZ_c
