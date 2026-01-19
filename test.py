import numpy as np
from spatial_process.fastbiliateral_blur import bilateral_filter, blur


from chromatic_adaptation import img_modified_CAT02_to_D65, img_CAT02_to_D65
from colour_space_conversion import XYZ_to_IPT, IPT_to_XYZ, XYZ_to_P3_RGB, XYZ_to_sRGB
from tone_compression.TC import img_TC
from colour_space_conversion.IPT_adjust import IPT_adjust
from PIL import Image

import datetime
import os

import scipy.io


def load_mat_file(file_path):
    """
    从 .mat 文件中加载数据。
    Args:
        file_path (str): .mat 文件的路径。
    Returns:
        dict: 加载的数据。
    """
    data = scipy.io.loadmat(file_path)
    return data


def create_folder_with_current_time():
    """
    创建一个以当前时间命名的文件夹。
    Returns:
        str: 新创建文件夹的路径。
    """
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join("example", folder_name)
    os.makedirs(
        folder_path, exist_ok=True
    )  # exist_ok=True 表示如果文件夹已存在，不会报错
    return folder_path


def save_image_temp(np_array, path):
    """
    将 numpy 数组保存为图像文件。
    Args:
        np_array (numpy.ndarray): 要保存的图像数据。
        path (str): 保存路径。
    """
    img = np_array / (np.max(np_array) + 1e-10)  # 归一化到 [0, 1]
    img = XYZ_to_sRGB(img)  # 转换到 sRGB
    img = img * 255  # 转换到 [0, 255]
    img = Image.fromarray(img.astype(np.uint8))
    img.save(path)


def LocalContrast(detail, base):
    La = 0.2 * base[:, :, 1]
    k = 1.0 / (5 * La + 1)
    FL = 0.2 * k**4 * (5 * La) + 0.1 * (1 - k**4) ** 2 * (5 * La) ** (1 / 3)
    FL_rep = np.stack([FL, FL, FL], axis=2)
    detail_a = detail ** ((FL_rep + 0.8) ** 0.25)
    return detail_a


def main():
    output_folder = create_folder_with_current_time()
    output_path = os.path.join(output_folder, "output.jpg")
    DETAIL = False
    index = 1

    # Input of the iCAM06 model: XYZ, absolute color space
    XYZ = np.load("example/xyz.npy").astype(np.float32)
    save_image_temp(XYZ, os.path.join(output_folder, f"{index:02d}_input.jpg"))
    index += 1

    # Image decomposition
    if DETAIL:
        base_layer, detail_layer = bilateral_filter(XYZ)
    else:
        base_layer = XYZ
        detail_layer = np.zeros_like(XYZ)
    save_image_temp(base_layer, os.path.join(output_folder, f"{index:02d}_base_layer.jpg"))
    index += 1
    save_image_temp(detail_layer, os.path.join(output_folder, f"{index:02d}_detail_layer.jpg"))
    index += 1

    # Chromatic adaptation
    white = blur(XYZ, 2)
    XYZ_adapt = img_CAT02_to_D65(base_layer, white, surround="average")
    save_image_temp(white, os.path.join(output_folder, f"{index:02d}_white_CAT.jpg"))
    index += 1
    save_image_temp(XYZ_adapt, os.path.join(output_folder, f"{index:02d}_XYZ_adapt.jpg"))
    index += 1

    # Tone compression
    white = blur(XYZ, 3)
    XYZ_tc = img_TC(XYZ_adapt, white, 0.7)
    save_image_temp(white, os.path.join(output_folder, f"{index:02d}_white_TC.jpg"))
    index += 1
    save_image_temp(XYZ_tc, os.path.join(output_folder, f"{index:02d}_XYZ_tc.jpg"))
    index += 1

    # Image attribute adjustments
    if DETAIL:
        XYZ_d = XYZ_tc * LocalContrast(detail_layer, base_layer)
    else:
        XYZ_d = XYZ_tc
    save_image_temp(XYZ_d, os.path.join(output_folder, f"{index:02d}_XYZ_d.jpg"))
    index += 1
    IPT = XYZ_to_IPT(XYZ_d)
    IPT_adjusted = IPT_adjust(IPT, XYZ_d)
    XYZ_p = IPT_to_XYZ(IPT_adjusted)

    # Convert XYZ to Display P3 RGB
    RGB_p = XYZ_to_P3_RGB(XYZ_p)
    RGB_p_uint8 = (RGB_p * 255).clip(0, 255).astype(np.uint8)

    # Create PIL Image and save
    img = Image.fromarray(RGB_p_uint8)
    with open("ICC/Display P3.icc", "rb") as icc_file:
        icc_profile = icc_file.read()
    img.save(output_path, icc_profile=icc_profile)
    print(f"Image saved as {output_path} with ICC profile.")


if __name__ == "__main__":
    # file_name = "XYZ.mat"
    # data = load_mat_file(file_name)
    # img1 = data["XYZimg"]
    # np.save("example/hdr2_float64.npy", img1)
    main()