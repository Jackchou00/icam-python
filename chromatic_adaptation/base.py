import numpy as np


class CATBase:
    """色适应转换基类"""

    def __init__(self, matrix, name="CAT"):
        self.matrix = matrix
        self.matrix_inv = np.linalg.inv(matrix)
        self.name = name
        self.xyz_wr = np.array([95.05, 100.0, 108.88])
        self.lms_wr = self.matrix @ self.xyz_wr

    def calc_D(self, XYZ_w, surround="average"):
        F = self._get_F(surround)
        La = 0.2 * XYZ_w[..., 1]
        D = F * (1 - (1 / 3.6) * np.exp(-(La + 42) / 92))
        return D

    def _get_F(self, surround):
        surround_dict = {"average": 1.0, "dim": 0.9, "dark": 0.8}
        return surround_dict.get(surround, 1.0)

    def transform(self, XYZ, XYZ_w, surround="average"):
        """执行色适应转换"""
        D = self.calc_D(XYZ_w, surround)

        # 转换到LMS空间
        XYZ_reshape = XYZ.reshape((-1, 3))
        LMS = XYZ_reshape @ self.matrix.T
        LMS = LMS.reshape(XYZ.shape)

        XYZ_w_reshape = XYZ_w.reshape((-1, 3))
        LMS_w = XYZ_w_reshape @ self.matrix.T
        LMS_w = LMS_w.reshape(XYZ_w.shape)

        # Yw / Ywr
        Yw_Ywr = XYZ_w[..., 1] / self.xyz_wr[1]

        # 应用色适应
        LMS_c = np.zeros_like(LMS)
        for i in range(3):
            LMS_c[..., i] = (D * Yw_Ywr * self.lms_wr[i] / LMS_w[..., i] + 1 - D) * LMS[
                ..., i
            ]

        # 转回XYZ空间
        LMS_c_reshape = LMS_c.reshape((-1, 3))
        XYZ_c = LMS_c_reshape @ self.matrix_inv.T
        XYZ_c = XYZ_c.reshape(XYZ.shape)

        return XYZ_c
