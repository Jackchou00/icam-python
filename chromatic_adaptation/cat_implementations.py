from .base import CATBase
import numpy as np


class CAT02(CATBase):
    def __init__(self):
        matrix = np.array(
            [
                [0.7328, 0.4296, -0.1624],
                [-0.7036, 1.6975, 0.0061],
                [0.0030, 0.0136, 0.9834],
            ]
        )
        super().__init__(matrix, "CAT02")


class CAT16(CATBase):
    def __init__(self):
        matrix = np.array(
            [
                [0.401288, 0.650173, -0.051461],
                [-0.250268, 1.204414, 0.045854],
                [-0.002079, 0.048952, 0.953127],
            ]
        )
        super().__init__(matrix, "CAT16")


class ModifiedCAT02(CAT02):
    def calc_D(self, XYZ_w, surround="average"):
        F = self._get_F(surround)
        La = 0.2 * XYZ_w[..., 1]
        # 注意这里的错误公式：(La - 42)而不是(La + 42)
        D = 0.3 * F * (1 - (1 / 3.6) * np.exp(-(La - 42) / 92))
        return D
