from typing import Tuple

import numpy as np
from numpy import linalg
from skimage import transform


def find_corners(mask) -> np.ndarray:
    """
    Finds corners of a rectangle which has undergone a projective transform.
    """
    y, x = np.where(mask)
    c = np.vstack((y, x)).T
    sorted_rows = np.argsort(c[:, 0])
    sorted_cols = np.argsort(c[:, 1])
    corners = np.array([
        c[sorted_cols[0]],
        c[sorted_rows[0]],
        c[sorted_cols[-1]],
        c[sorted_rows[-1]]
    ])
    return corners


def compute_plane_size(corners: np.ndarray, coords: np.ndarray) -> Tuple[int, int]:
    corner_coords = np.vstack([
        coords[corners[:,0], corners[:, 1], 0],
        coords[corners[:,0], corners[:, 1], 1],
        coords[corners[:,0], corners[:, 1], 2]
    ])

    width = linalg.norm(corner_coords[:, 0] - corner_coords[:, 1])
    height = linalg.norm(corner_coords[:, 1] - corner_coords[:, 2])
    return height, width


def rectify_plane(image, corners, height, width):
    src = np.array((
            (0, 0),
            (width, 0),
            (width, height),
            (0, height),
        ))
    dst = corners[:, [1, 0]]
    tform = transform.ProjectiveTransform()
    tform.estimate(src, dst)
    im_min, im_max = image.min(), image.max()
    rectified = transform.warp((image - im_min)/(im_max - im_min), tform,
                               output_shape=(height, width))[:,:]
    rectified = rectified * (im_max - im_min) + im_min
    return rectified
