import numpy as np
from PIL import Image


M = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


def XYZ_to_sRGB(XYZ):
    """
    Convert XYZ color space to sRGB color space.

    Args:
        XYZ (np.ndarray): Input image in XYZ color space.

    Returns:
        np.ndarray: Converted image in sRGB color space.
    """
    # normalize XYZ values to [0, 1]
    XYZ = np.clip(XYZ / 100, 0, 1)

    # Apply the transformation
    XYZ_reshaped = XYZ.reshape(-1, 3)
    RGB_reshaped = np.dot(XYZ_reshaped, M.T)
    RGB = RGB_reshaped.reshape(XYZ.shape)

    # Apply the sRGB EOTF inverse
    RGB = np.where(RGB <= 0.0031308, 12.92 * RGB, 1.055 * (RGB ** (1 / 2.4)) - 0.055)

    # Clip values to [0, 1] range
    RGB = np.clip(RGB, 0, 1)
    return RGB


def save_image(XYZ, path):
    """
    Save the image to the specified path.

    Args:
        XYZ (np.ndarray): Image to save.
        path (str): Path to save the image.
    """
    # Convert to sRGB
    RGB = XYZ_to_sRGB(XYZ)

    # Convert to uint8, if saved as JPEG, save with quality 100
    image = (RGB * 255).astype(np.uint8)
    img = Image.fromarray(image)
    if path.lower().endswith(".jpg"):
        img.save(path, quality=100)
    else:
        img.save(path)
