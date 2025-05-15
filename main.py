import numpy as np
import icam06
from image_input import read_img
import os


def main():
    output_folder = "example"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    print("Step 1: Reading input image...")
    # Input of the iCAM06 model: XYZ, absolute color space
    # XYZ = np.load("example/xyz.npy").astype(np.float32)
    file_name = "test_0.tif"
    XYZ = read_img(file_name).astype(np.float32)

    # save a original image as comparison
    # original image: normalized by max Y value, linear output.
    print("Step 2: Saving original image...")
    icam06.save_image(
        XYZ * 100 / np.max(XYZ[..., 1]), f"{output_folder}/original_image.jpg"
    )

    # Image decomposition
    print("Step 3: Performing image decomposition (bilateral filtering)...")
    base_layer, detail_layer = icam06.bilateral_filter(XYZ)
    icam06.save_image(
        base_layer * 100 / np.max(base_layer[..., 1]), f"{output_folder}/base_layer.jpg"
    )
    icam06.save_image(
        detail_layer * 100 / np.max(detail_layer[..., 1]),
        f"{output_folder}/detail_layer.jpg",
    )

    # Chromatic adaptation
    print("Step 4: Calculating white point for chromatic adaptation...")
    white = icam06.blur(XYZ, 2)
    icam06.save_image(
        white * 100 / np.max(white[..., 1]), f"{output_folder}/white_adaptation.jpg"
    )

    # here we offer two options for chromatic adaptation:
    # 1. CAT02 chromatic adaptation with fixed calculation of D and corresponding color.
    # 2. original one in paper, which may lead to numerical issue.
    print("Step 5: Performing chromatic adaptation...")
    # XYZ_adapt = icam06.CAT02_to_D65_fixed(base_layer, white, surround="average")
    XYZ_adapt = icam06.CAT02_to_D65(base_layer, white, surround="average")
    icam06.save_image(
        XYZ_adapt * 100 / np.max(XYZ_adapt[..., 1]), f"{output_folder}/XYZ_adapted.jpg"
    )

    # Tone compression
    print("Step 6: Calculating white point for tone compression...")
    white = icam06.blur(XYZ, 3)
    icam06.save_image(
        white * 100 / np.max(white[..., 1]), f"{output_folder}/white_compression.jpg"
    )

    print("Step 7: Performing tone compression...")
    XYZ_tc = icam06.tone_compression(XYZ_adapt, white, p=0.75)
    icam06.save_image(
        XYZ_tc * 100 / np.max(XYZ_tc[..., 1]),
        f"{output_folder}/XYZ_tone_compressed.jpg",
    )

    # Combine base and detail layers
    print("Step 8: Combining base and detail layers...")
    XYZ_d = icam06.combine(XYZ_tc, detail_layer)
    icam06.save_image(
        XYZ_d * 100 / np.max(XYZ_d[..., 1]), f"{output_folder}/XYZ_combined.jpg"
    )

    # Image attribute adjustments
    print("Step 9: Performing IPT adjustments...")
    XYZ_p = icam06.IPT_adjust(XYZ_d, surround="average")
    icam06.save_image(
        XYZ_p * 100 / np.max(XYZ_p[..., 1]), f"{output_folder}/XYZ_IPT_adjusted.jpg"
    )

    # Output image
    print("Step 10: Saving final output image...")
    output_file = f"{output_folder}/output.jpg"
    icam06.save_image(XYZ_p, output_file)
    print(f"Processing complete. All results saved to '{output_folder}' folder.")


if __name__ == "__main__":
    main()
