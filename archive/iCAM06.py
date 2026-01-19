import numpy as np
import numexpr as ne


def changeColorSpace(inImage, colorMatrix):
    shape = inImage.shape
    px = inImage.reshape((-1, 3))
    out = px @ colorMatrix
    outImage = out.reshape(shape)
    return outImage


def iCAM06_CAT(XYZimg, white):
    # transform the XYZ to RGB (sensor) space
    M = np.array(
        [[0.7328, -0.7036, 0.0030], [0.4296, 1.6974, 0.0136], [-0.1624, 0.0061, 0.9834]]
    )

    RGB_img = changeColorSpace(XYZimg, M)
    RGB_white = changeColorSpace(white, M)
    xyz_d65 = np.array([95.05, 100.0, 108.88])
    RGB_d65 = (xyz_d65.T @ M).T

    La = 0.2 * white[..., 1]
    F = 1.0
    D = ne.evaluate("0.3 * F * (1 - (1 / 3.6) * exp(-(La - 42) / 92))") ## 似乎应该是 exp(-(La + 42) / 92)

    RGB_white = RGB_white + 1e-7
    Rc = (D * RGB_d65[0] / RGB_white[..., 0] + 1 - D) * RGB_img[..., 0]
    Gc = (D * RGB_d65[1] / RGB_white[..., 1] + 1 - D) * RGB_img[..., 1]
    Bc = (D * RGB_d65[2] / RGB_white[..., 2] + 1 - D) * RGB_img[..., 2]

    adaptImage = np.stack([Rc, Gc, Bc], axis=-1)
    XYZ_adapt = changeColorSpace(adaptImage, np.linalg.inv(M))
    return XYZ_adapt


def iCAM06_TC(XYZ_adapt, white_img, p):
    # transform the adapted XYZ to Hunt-Pointer-Estevez space
    M = np.array(
        [[0.38971, -0.22981, 0.0], [0.68898, 1.18340, 0.0], [-0.07868, 0.04641, 1.0]]
    )
    Mi = np.linalg.inv(M)
    RGB_img = changeColorSpace(XYZ_adapt, M)

    # cone response
    La = 0.2 * white_img[..., 1]
    k = 1.0 / (5 * La + 1)
    k4 = ne.evaluate("k ** 4")  # 预计算 k^4
    FL = ne.evaluate("0.2 * k4 * (5 * La) + 0.1 * (1 - k4) ** 2 * (5 * La) ** (1/3)")

    # compression
    sign_RGB = np.sign(RGB_img)
    FL_rep = FL[:, :, np.newaxis]  # 使用广播
    white_img_rep = white_img[:, :, 1][:, :, np.newaxis]  # 使用广播

    # 计算中间表达式
    ratio = ne.evaluate("FL_rep * abs(RGB_img) / white_img_rep")
    term1 = ne.evaluate("400 * ratio ** p")
    term2 = ne.evaluate("27.13 + ratio ** p")

    # 计算最终结果
    RGB_c = ne.evaluate("sign_RGB * (term1 / term2) + 0.1")

    # make a neutral As Rod response
    Las = 2.26 * La
    j = 0.00001 / (5 * Las / 2.26 + 0.00001)
    j2 = ne.evaluate("j ** 2")  # 预计算 j^2
    FLS = ne.evaluate(
        "3800 * j2 * (5 * Las / 2.26) + 0.2 * (1 - j2) ** 4 * (5 * Las / 2.26) ** (1/6)"
    )
    Sw = np.max(5 * La)

    # 计算 S
    S = np.abs(np.repeat(np.abs(XYZ_adapt[:, :, 1])[:, :, np.newaxis], 3, axis=2))

    # 计算 Bs
    Las_rep = Las[:, :, np.newaxis]  # 使用广播
    ratio2 = ne.evaluate("(5 * Las_rep / 2.26) * (S / Sw)")
    term1 = ne.evaluate("0.5 / (1 + 0.3 * ratio2 ** 3)")  # 似乎应该是 ** 0.3
    term2 = ne.evaluate("0.5 / (1 + 5 * (5 * Las_rep / 2.26))")
    Bs = term1 + term2

    # 计算 As
    FLS_rep = FLS[:, :, np.newaxis]  # 使用广播
    ratio3 = ne.evaluate("FLS_rep * (S / Sw)")
    term3 = ne.evaluate("400 * ratio3 ** p")
    term4 = ne.evaluate("27.13 + ratio3 ** p")
    As = ne.evaluate("3.05 * Bs * (term3 / term4) + 0.03") # 似乎应该是 + 0.3

    # combine Cone and Rod response
    RGB_c = RGB_c + As

    # convert RGB_c back to XYZ space
    XYZ_tc = changeColorSpace(RGB_c, Mi)
    return XYZ_tc


def iCAM06_IPT(XYZ_img, base_img, gamma):
    # transform into IPT space
    xyz2lms = np.array(
        [[0.4002, 0.7077, -0.0807], [-0.2280, 1.1500, 0.0612], [0.0000, 0.0000, 0.9184]]
    ).T

    iptMat = np.array(
        [[0.4000, 0.4000, 0.2000], [4.4550, -4.8510, 0.3960], [0.8056, 0.3572, -1.1628]]
    ).T

    # convert to LMS space
    lms_img = changeColorSpace(XYZ_img, xyz2lms)

    # lms_nonlinear_img = np.sign(lms_img) * np.abs(lms_img) ** 0.43
    abs_lms_pow = ne.evaluate("abs(lms_img) ** 0.43")
    sign_lms = np.sign(lms_img)
    lms_nonlinear_img = ne.evaluate("sign_lms * abs_lms_pow")

    # apply the IPT exponent
    ipt_img = changeColorSpace(lms_nonlinear_img, iptMat)

    # colorfulness adjustment - Hunt effect
    c = np.sqrt(ipt_img[:, :, 1] ** 2 + ipt_img[:, :, 2] ** 2)
    La = 0.2 * base_img[:, :, 1]
    k = 1 / (5 * La + 1)
    k4 = ne.evaluate("k ** 4")
    FL = ne.evaluate("0.2 * k4 * (5 * La) + 0.1 * (1 - k4) ** 2 * (5 * La) ** (1 / 3)")

    c2 = ne.evaluate("c ** 2")
    adjustment = ne.evaluate("(FL + 1) ** 0.15 * ((1.29 * c2 - 0.27 * c + 0.42) / (c2 - 0.31 * c + 0.42))")
    ipt_img[:, :, 1] = ipt_img[:, :, 1] * adjustment
    ipt_img[:, :, 2] = ipt_img[:, :, 2] * adjustment

    # Bartleson surround adjustment
    max_i = np.max(ipt_img[:, :, 0])
    ipt_img[:, :, 0] = ipt_img[:, :, 0] / max_i
    ipt_img[:, :, 0] = ipt_img[:, :, 0] ** gamma
    ipt_img[:, :, 0] = ipt_img[:, :, 0] * max_i

    # inverse IPT
    lms_nonlinear_img = changeColorSpace(ipt_img, np.linalg.inv(iptMat))

    lms_img = np.sign(lms_nonlinear_img) * np.abs(lms_nonlinear_img) ** (1 / 0.43)

    XYZ_p = changeColorSpace(lms_img, np.linalg.inv(xyz2lms))

    return XYZ_p


def iCAM06_invcat(XYZ_img):
    M = np.array(
        [
            [0.8562, 0.3372, -0.1934],
            [-0.8360, 1.8327, 0.0033],
            [0.0357, -0.0469, 1.0112],
        ]
    )
    Mi = np.linalg.inv(M)
    RGB_img = changeColorSpace(XYZ_img, M)
    XYZ_adapt = changeColorSpace(RGB_img, Mi)
    return XYZ_adapt
