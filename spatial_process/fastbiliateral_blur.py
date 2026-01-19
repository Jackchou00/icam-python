import numpy as np
import numexpr as ne
import cv2


def idl_dist(m, n):
    """
    Compute a 2D Euclidean distance map similar to IDL's DIST function.

    Parameters:
    m (int): Number of rows in the output matrix.
    n (int): Number of columns in the output matrix.

    Returns:
    numpy.ndarray: A 2D matrix where each element represents the Euclidean distance from the center.
    """
    y, x = np.ogrid[:m, :n]
    return np.sqrt(np.minimum(x, n - 1 - x)**2 + np.minimum(y, m - 1 - y)**2)


def blur(img, d):
    """
    Generate a low-pass version of the input image as adapted white.

    Parameters:
    img (numpy.ndarray): Input image (3D array, shape: [height, width, channels]).
    d (float): Parameter controlling the Gaussian kernel size.

    Returns:
    numpy.ndarray: Low-pass filtered image (adapted white).
    """
    # Downsampling
    sy, sx, sz = img.shape
    m = min(sy, sx)
    if m < 64:
        z = 1
    elif m < 256:
        z = 2
    elif m < 512:
        z = 4
    elif m < 1024:
        z = 8
    elif m < 2056:
        z = 16
    else:
        z = 32
    img = img[::z, ::z, :]  # Downsample the image
    # Get the size of the downsampled image
    yDim, xDim, _ = img.shape
    # Padding with symmetric mirroring
    Y = np.zeros((2 * yDim, 2 * xDim, 3))
    # Center
    Y[yDim//2:yDim//2 + yDim, xDim//2:xDim//2 + xDim, :] = img
    # Left mirror
    Y[yDim//2:yDim//2 + yDim, :xDim//2, :] = img[:, :xDim//2, :][:, ::-1, :]
    # Right mirror
    Y[yDim//2:yDim//2 + yDim, xDim//2 + xDim:, :] = img[:, xDim//2:xDim, :][:, ::-1, :]
    # Top mirror
    Y[:yDim//2, xDim//2:xDim//2 + xDim, :] = img[:yDim//2, :, :][::-1, :, :]
    # Bottom mirror
    Y[yDim//2 + yDim:, xDim//2:xDim//2 + xDim, :] = img[yDim//2:, :, :][::-1, :, :]
    # Top-left corner
    Y[:yDim//2, :xDim//2, :] = img[:yDim//2, :xDim//2, :][::-1, ::-1, :]
    # Top-right corner
    Y[:yDim//2, xDim//2 + xDim:, :] = img[:yDim//2, xDim//2:xDim, :][::-1, ::-1, :]
    # Bottom-right corner
    Y[yDim//2 + yDim:, xDim//2 + xDim:, :] = img[yDim//2:, xDim//2:xDim, :][::-1, ::-1, :]
    # Bottom-left corner
    Y[yDim//2 + yDim:, :xDim//2, :] = img[yDim//2:, :xDim//2, :][::-1, ::-1, :]
    # Gaussian Filtering
    dist_map = idl_dist(Y.shape[0], Y.shape[1])
    Dim = max(xDim, yDim)
    kernel = np.exp(-1 * (dist_map / (Dim / d)) ** 2)
    filter_kernel = np.maximum(np.real(np.fft.fft2(kernel)), 0)
    filter_kernel = filter_kernel / filter_kernel[0, 0]
    # Apply the filter to each channel
    white = np.zeros_like(Y)
    for channel in range(3):
        white[:, :, channel] = np.maximum(
            np.real(np.fft.ifft2(np.fft.fft2(Y[:, :, channel]) * filter_kernel)),
            0
        )
    # Crop the padded image
    white = white[yDim//2:yDim//2 + yDim, xDim//2:xDim//2 + xDim, :]
    # Upsampling
    white = cv2.resize(white, (sx, sy), interpolation=cv2.INTER_NEAREST)
    return white


def bilateral_filter(img):
    img[img < 1e-4] = 1e-4
    logimg = ne.evaluate("log10(img)")

    sigmaColor = 0.35
    if min(img.shape) < 1024:
        z = 2
    else:
        z = 4
    _, xDim, _ = img.shape
    # sigmaSpace = 2 * xDim / z / 100
    sigmaSpace = 10
    base_layer = cv2.bilateralFilter(logimg, d=-1, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    
    detail_layer = logimg - base_layer
    detail_layer[detail_layer > 12] = 0
    
    base_layer = ne.evaluate("10**base_layer")
    detail_layer = ne.evaluate("10**detail_layer")

    return base_layer, detail_layer