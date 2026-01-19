import numpy as np


M_CAT16 = np.array(
    [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ]
)
M_CAT16_inv = np.linalg.inv(M_CAT16)
XYZ_wr = np.array([95.05, 100.0, 108.88])
LMS_wr = M_CAT16 @ XYZ_wr


def img_CAT16_to_D65(XYZ, XYZ_w, surround="average"):
    # XYZ input: shape (h, w, 3)
    # XYZ_w: shape (h, w, 3)
    # D: shape (h, w)
    D = calc_D(XYZ_w, surround)
    
    
    XYZ_reshape = XYZ.reshape((-1, 3))
    LMS = XYZ_reshape @ M_CAT16.T
    LMS = LMS.reshape(XYZ.shape)

    XYZ_w_reshape = XYZ_w.reshape((-1, 3))
    LMS_w = XYZ_w_reshape @ M_CAT16.T
    LMS_w = LMS_w.reshape(XYZ_w.shape)
    
    
    # Yw / Ywr
    Yw_Ywr = XYZ_w[... , 1] / XYZ_wr[1]
    
    
    LMS_c = np.zeros_like(LMS)
    for i in range(3):
        LMS_c[..., i] = (D * Yw_Ywr * LMS_wr[i] / LMS_w[..., i] + 1 - D) * LMS[..., i]


    # reshape LMS to (3, h*w)
    LMS_c_reshape = LMS_c.reshape((-1, 3))
    XYZ_c = LMS_c_reshape @ M_CAT16_inv.T
    XYZ_c = XYZ_c.reshape(XYZ.shape)

    return XYZ_c


def get_F(surround):
    surround_dict = {
        "average": 1.0,
        "dim": 0.9,
        "dark": 0.8
    }
    return surround_dict.get(surround, 1.0)  # default to 1.0 if surround is not found
    
    
def calc_D(XYZ_w, surround):
    F = get_F(surround)
    La = 0.2 * XYZ_w[..., 1]
    D = F * (1 - (1 / 3.6) * np.exp(-(La + 42) / 92))
    return D