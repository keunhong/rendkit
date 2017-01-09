import numpy as np
from numpy import linalg


def normalize_to_range(array, lo, hi):
    if hi <= lo:
        raise ValueError('Range must be increasing but {} >= {}.'.format(
            lo, hi))
    min_val = array.min()
    max_val = array.max()
    scale = max_val - min_val if (min_val < max_val) else 1
    return (array - min_val) / scale * (hi - lo) + lo


def normalize_to_unit(points):
    return points / linalg.norm(points, axis=1)[:, np.newaxis]


def filter_outliers(data, k=2):
    data = np.array(data)
    inlier_inds = np.abs(data - np.mean(data)) < k * np.std(data)
    return data[inlier_inds]


def robust_mean(data, k=2):
    return np.mean(filter_outliers(data, k=k))
