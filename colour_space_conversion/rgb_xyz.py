import numpy as np
from typing import Literal, Union, List, Dict, Any, Tuple, TypedDict


ColorSpaceType = Literal["srgb", "display-p3", "dci-p3-63", "bt2020"]
TransferFunctionType = Literal["srgb", "linear", "gamma"]


class PrimaryCoordinates(TypedDict):
    red: Tuple[float, float]
    green: Tuple[float, float]
    blue: Tuple[float, float]
    white: Tuple[float, float]


ColorSpaceInput = Union[ColorSpaceType, PrimaryCoordinates]


class TransferFunctionParams(TypedDict, total=False):
    type: TransferFunctionType
    gamma: float


def get_available_color_spaces() -> List[str]:
    """
    Returns a list of all available color space names.

    Returns:
        List[str]: List of available color space names
    """
    return list(COLOR_SPACES.keys())


def RGB_to_XYZ(
    RGB,
    space: ColorSpaceInput = "srgb",
    transfer_function: Union[TransferFunctionType, TransferFunctionParams] = "srgb",
):
    """
    Converts RGB values to XYZ values.

    Args:
        RGB: Array of RGB values
        space: Color space name or custom primaries dictionary. Options include:
               - "srgb" (default): sRGB color space
               - "display-p3": Display P3 color space
               - "dci-p3-63": DCI-P3 (Theatrical) color space
               - "bt2020": BT.2020 color space
               - Custom dictionary with red, green, blue, and white point coordinates
        transfer_function: Transfer function type or parameters dictionary

    Returns:
        np.ndarray: Converted XYZ values
    """
    # linear RGB
    linear_RGB = eotf(RGB, transfer_function)
    # Convert to XYZ
    ccm = generate_ccm(space)
    XYZ = linear_RGB @ ccm.T
    return XYZ


def XYZ_to_RGB(
    XYZ,
    space: ColorSpaceInput = "srgb",
    transfer_function: Union[TransferFunctionType, TransferFunctionParams] = "srgb",
):
    """
    Converts XYZ values to RGB values.

    Args:
        XYZ: Array of XYZ values
        space: Color space name or custom primaries dictionary. Options include:
               - "srgb" (default): sRGB color space
               - "display-p3": Display P3 color space
               - "dci-p3-63": DCI-P3 (Theatrical) color space
               - "bt2020": BT.2020 color space
               - Custom dictionary with red, green, blue, and white point coordinates
        transfer_function: Transfer function type or parameters dictionary

    Returns:
        np.ndarray: Converted RGB values
    """
    # Generate the color conversion matrix
    ccm = generate_ccm(space)
    # Calculate inverse CCM to go from XYZ to linear RGB
    inv_ccm = np.linalg.inv(ccm)
    # Convert to linear RGB
    linear_RGB = XYZ @ inv_ccm.T
    # Apply OETF to get display-referred RGB
    RGB = oetf(linear_RGB, transfer_function)
    return RGB


def eotf(
    RGB: np.ndarray, tf_params: Union[TransferFunctionType, TransferFunctionParams]
) -> np.ndarray:
    """
    Applies Electro-Optical Transfer Function (EOTF) to convert display-referred RGB to scene-referred linear RGB.

    Args:
        RGB: Display-referred RGB values
        tf_params: Transfer function type ("srgb", "linear", "gamma") or parameters dictionary
                  When using "gamma", you can specify the gamma value in the parameters

    Returns:
        np.ndarray: Scene-referred linear RGB values
    """
    if isinstance(tf_params, str):
        tf_type = tf_params
        params = {}
    else:
        tf_type = tf_params["type"]
        params = tf_params

    if tf_type == "srgb":
        linear_RGB = np.where(
            RGB <= 0.04045, RGB / 12.92, ((RGB + 0.055) / 1.055) ** 2.4
        )
    elif tf_type == "linear":
        linear_RGB = RGB
    elif tf_type == "gamma":
        gamma = params.get("gamma", 2.2)
        linear_RGB = RGB**gamma
    else:
        raise ValueError("Unsupported transfer function")
    return linear_RGB


def oetf(
    linear_RGB: np.ndarray,
    tf_params: Union[TransferFunctionType, TransferFunctionParams],
) -> np.ndarray:
    """
    Applies Opto-Electronic Transfer Function (OETF) to convert scene-referred linear RGB to display-referred RGB.

    Args:
        linear_RGB: Scene-referred linear RGB values
        tf_params: Transfer function type ("srgb", "linear", "gamma") or parameters dictionary
                  When using "gamma", you can specify the gamma value in the parameters

    Returns:
        np.ndarray: Display-referred RGB values
    """
    if isinstance(tf_params, str):
        tf_type = tf_params
        params = {}
    else:
        tf_type = tf_params["type"]
        params = tf_params

    if tf_type == "srgb":
        RGB = np.where(
            linear_RGB <= 0.0031308,
            12.92 * linear_RGB,
            1.055 * (linear_RGB ** (1 / 2.4)) - 0.055,
        )
    elif tf_type == "linear":
        RGB = linear_RGB
    elif tf_type == "gamma":
        gamma = params.get("gamma", 2.2)
        RGB = linear_RGB ** (1 / gamma)
    else:
        raise ValueError("Unsupported transfer function")
    return RGB


# Define the color spaces and their corresponding primaries and white points
COLOR_SPACES = {
    "srgb": {
        "primaries": [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
        "white_point": [0.3127, 0.3290],
    },
    "display-p3": {
        "primaries": [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        "white_point": [0.3127, 0.3290],
    },
    "dci-p3-63": {
        "primaries": [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        "white_point": [0.314, 0.351],
    },
    "bt2020": {
        "primaries": [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        "white_point": [0.3127, 0.3290],
    },
}


def generate_ccm(space: ColorSpaceInput) -> np.ndarray:
    """
    Generates a Color Conversion Matrix (CCM) for the specified color space.

    Args:
        space: Color space name or custom primaries dictionary.

    Returns:
        np.ndarray: Color conversion matrix from RGB to XYZ

    Raises:
        ValueError: When the specified color space is not supported
    """
    if isinstance(space, dict):
        primaries = [space["red"], space["green"], space["blue"]]
        white_point = space["white"]
    else:
        if space not in COLOR_SPACES:
            raise ValueError(f"Unsupported color space: {space}")
        color_space_data = COLOR_SPACES[space]
        primaries = color_space_data["primaries"]
        white_point = color_space_data["white_point"]

    xys = primaries + [white_point]

    # assume all Y values are 1.0
    def xy_to_xyz(x, y, Y):
        X = (x * Y) / y
        Z = ((1 - x - y) * Y) / y
        return np.array([X, Y, Z])

    XYZs = []
    for xy in xys:
        XYZ = xy_to_xyz(xy[0], xy[1], 1.0)
        XYZs.append(XYZ)

    XYZ_w = np.array(XYZs[-1])
    XYZs = np.array(XYZs[:-1])

    ratio = np.linalg.solve(XYZs.T, XYZ_w.T).T
    ccm = XYZs.T @ np.diag(ratio)
    return ccm


def main():
    # Example usage of the RGB_to_XYZ function with a custom color space
    custom_space = {
        "red": (0.64, 0.33),
        "green": (0.30, 0.60),
        "blue": (0.15, 0.06),
        "white": (0.3127, 0.3290),
    }

    rgb = np.array([1.0, 1.0, 1.0])  # Example RGB value (white)
    xyz = RGB_to_XYZ(rgb, space=custom_space, transfer_function="srgb")
    print("RGB:", rgb)
    print("XYZ:", xyz)
    
    # Example of round-trip conversion
    rgb_back = XYZ_to_RGB(xyz, space=custom_space, transfer_function="srgb")
    print("RGB (round-trip):", rgb_back)


if __name__ == "__main__":
    main()
