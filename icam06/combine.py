import numpy as np


def local_contrast(base_layer, detail_layer):
    La = 0.2 * base_layer[:, :, 1]
    k = 1.0 / (5 * La + 1)
    FL = 0.2 * k**4 * (5 * La) + 0.1 * (1 - k**4) ** 2 * (5 * La) ** (1 / 3)
    FL_rep = np.stack([FL, FL, FL], axis=2)
    detail_a = detail_layer ** ((FL_rep + 0.8) ** 0.25)
    return detail_a


def combine(base_layer, detail_layer):
    """
    Combine the base and detail layers using a weighted sum.

    Args:
        base_layer (np.ndarray): Base layer image.
        detail_layer (np.ndarray): Detail layer image.

    Returns:
        np.ndarray: Combined image.
    """
    combined = base_layer * local_contrast(base_layer, detail_layer)
    return combined
