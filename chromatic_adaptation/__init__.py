from .CAT_vK20 import img_vK20_to_D65
from .cat_implementations import CAT02, CAT16, ModifiedCAT02

# 创建实例
cat02 = CAT02()
cat16 = CAT16()
modified_cat02 = ModifiedCAT02()


# 导出函数
def img_CAT02_to_D65(XYZ, XYZ_w, surround="average"):
    return cat02.transform(XYZ, XYZ_w, surround)


def img_CAT16_to_D65(XYZ, XYZ_w, surround="average"):
    return cat16.transform(XYZ, XYZ_w, surround)


def img_modified_CAT02_to_D65(XYZ, XYZ_w, surround="average"):
    return modified_cat02.transform(XYZ, XYZ_w, surround)
