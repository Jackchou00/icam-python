import numpy as np
import icam06


def main():
    output_path = "example/output_cat_fixed.jpg"

    # Input of the iCAM06 model: XYZ, absolute color space
    XYZ = np.load("example/xyz.npy").astype(np.float32)

    # save a original image as comparison
    # original image: normalized by max Y value, linear output.
    icam06.save_image(XYZ * 100 / np.max(XYZ[..., 1]), "example/original_image.jpg")

    # Image decomposition
    base_layer, detail_layer = icam06.bilateral_filter(XYZ)

    # Chromatic adaptation
    white = icam06.blur(XYZ, 2)
    # here we offer two options for chromatic adaptation:
    # 1. CAT02 chromatic adaptation with fixed calculation of D and corresponding color.
    # 2. original one in paper, which may lead to numerical issue.
    
    # XYZ_adapt = icam06.CAT02_to_D65_fixed(base_layer, white, surround="average")
    XYZ_adapt = icam06.CAT02_to_D65(base_layer, white, surround="average")

    # Tone compression
    white = icam06.blur(XYZ, 3)
    XYZ_tc = icam06.tone_compression(XYZ_adapt, white, p=0.75)

    # Combine base and detail layers
    XYZ_d = icam06.combine(XYZ_tc, detail_layer)

    # Image attribute adjustments
    XYZ_p = icam06.IPT_adjust(XYZ_d, surround="average")

    # Output image
    icam06.save_image(XYZ_p, output_path)


if __name__ == "__main__":
    main()
