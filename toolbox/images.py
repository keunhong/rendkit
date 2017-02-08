import logging
import math
from typing import Tuple

import numpy as np
from scipy import misc
from scipy.ndimage.interpolation import zoom
from skimage import transform
from toolbox.stats import find_outliers

logger = logging.getLogger(__name__)


BoundingBox = Tuple[int, int, int, int]


def compute_segment_median_colors(image: np.ndarray, segment_mask: np.ndarray):
    num_segments = sum((1 for i in np.unique(segment_mask) if i >= 0))

    segment_colors = []
    for segment_id in range(num_segments):
        segment_pixels = image[segment_mask == segment_id]
        if len(segment_pixels) > 0:
            median_color = np.median(segment_pixels, axis=0)
            segment_colors.append((median_color[0],
                                   median_color[1],
                                   median_color[2]))
            logger.info('segment {}: {} pixels, median_color={}'.format(
                segment_id, len(segment_pixels), repr(median_color)))
        else:
            segment_colors.append((1, 0, 1))
            logger.info('segment {}: not visible.'.format(segment_id))

    return np.array(segment_colors)


def compute_mask_bbox(mask: np.ndarray) -> BoundingBox:
    """
    Computes bounding box which contains the mask.
    :param mask:
    :return:
    """
    yinds, xinds = np.where(mask)
    xmin, xmax = np.min(xinds), np.max(xinds)
    ymin, ymax = np.min(yinds), np.max(yinds)

    return ymin, ymax, xmin, xmax


def crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    ymin, ymax, xmin, xmax = bbox
    return image[ymin:ymax, xmin:xmax]


def pad(image: np.ndarray,
        pad_width: int,
        mode='constant',
        fill=0) -> np.ndarray:
    if len(image.shape) == 4:
        padding = ((pad_width,), (pad_width,), (0,), (0,))
    elif len(image.shape) == 3:
        padding = ((pad_width,), (pad_width,), (0,))
    elif len(image.shape) == 2:
        padding = ((pad_width,), (pad_width,))
    else:
        raise RuntimeError("Unsupported image shape {}".format(image.shape))

    return np.pad(image, pad_width=padding, mode=mode, constant_values=fill)


def rotate(image: np.ndarray, angle: float, crop=False) -> np.ndarray:
    rotated = transform.rotate(image, angle)
    if crop:
        radius = min(image.shape[:2]) / 2.0
        length = math.sqrt(2) * radius
        height, width = image.shape[:2]
        rotated = rotated[
                  height/2-length/2:height/2+length/2,
                  width/2-length/2:width/2+length/2]
    return rotated


def apply_mask(image, mask, fill=0):
    """
    Fills pixels outside the mask with a constant value.
    :param image: to apply the mask to.
    :param mask: binary mask with True values for pixels that are to be preserved.
    :param fill: fill value.
    :return: Masked image
    """
    masked = image.copy()
    masked[~mask] = fill
    return masked


def normalize(image, low=0.0, high=1.0):
    """
    Normalized the image to a range.
    :param image:
    :param low: lowest value after normalization.
    :param high: highest value after normalization.
    :return:
    """
    image_01 = (image - image.min()) / (image.max() - image.min())
    return image_01 * (high - low) + low


def rgb2gray(image):
    return (0.2125 * image[:, :, 0]
            + 0.7154 * image[:, :, 1]
            + 0.0721 * image[:, :, 2])


def trim_image(image, mask):
    y, x = np.where(mask)
    return image[y.min():y.max(), x.min():x.max()]


def suppress_outliers(image):
    new_map = image.copy()
    outliers = find_outliers(np.reshape(image, (-1, 3))).reshape(image.shape[:2])
    med = np.median(image, axis=(0, 1))
    new_map[outliers] = med
    return new_map


def resize(array, shape, order=2):
    if len(array.shape) != 2 and len(array.shape) != 3:
        raise RuntimeError("Input array must have 2 or 3 dimensions but {} "
                           "were given.".format(len(array.shape)))
    if isinstance(shape, float):
        scales = (shape, shape, 1)
    elif isinstance(shape, tuple):
        scales = (shape[0] / array.shape[0],
                  shape[1] / array.shape[1], 1)
    else:
        raise RuntimeError("shape must be tuple or float.")

    n_channels = 1 if len(array.shape) == 2 else array.shape[2]
    if n_channels == 1:
        scales = scales[:2]
    output = zoom(array, scales, order=order)
    return output


def imsave(path, array):
    array = array.astype(dtype=float)
    if array.max() <= 1.0:
        array *= 255.0
    misc.toimage(array, cmin=0, cmax=255).save(path)