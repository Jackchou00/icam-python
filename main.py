import numpy as np
from spatial_process.fastbiliateral_blur import bilateral_filter, blur


from chromatic_adaptation import img_modified_CAT02_to_D65, img_CAT02_to_D65
from colour_space_conversion import XYZ_to_IPT, IPT_to_XYZ, XYZ_to_P3_RGB
from tone_compression.TC import img_TC
from colour_space_conversion.IPT_adjust import IPT_adjust
from PIL import Image


def LocalContrast(detail, base):
    La = 0.2 * base[:, :, 1]
    k = 1.0 / (5 * La + 1)
    FL = 0.2 * k**4 * (5 * La) + 0.1 * (1 - k**4) ** 2 * (5 * La) ** (1 / 3)
    FL_rep = np.stack([FL, FL, FL], axis=2)
    detail_a = detail ** ((FL_rep + 0.8) ** 0.25)
    return detail_a


def main():
    output_path = "example/output_no_detailed_P3.jpg"
    DETAIL = False

    # Input of the iCAM06 model: XYZ, absolute color space
    XYZ = np.load("example/xyz.npy").astype(np.float32)

    # Image decomposition
    if DETAIL:
        base_layer, detail_layer = bilateral_filter(XYZ)
    else:
        base_layer = XYZ

    # Chromatic adaptation
    white = blur(XYZ, 2)
    XYZ_adapt = img_CAT02_to_D65(base_layer, white, surround="average")

    # Tone compression
    white = blur(XYZ, 3)
    XYZ_tc = img_TC(XYZ_adapt, white, 0.7)

    # Image attribute adjustments
    if DETAIL:
        XYZ_d = XYZ_tc * LocalContrast(detail_layer, base_layer)
    else:
        XYZ_d = XYZ_tc
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
    main()
