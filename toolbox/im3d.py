import numpy as np
from numpy import linalg
from scipy.spatial import ConvexHull
from skimage import transform
from toolbox.data import reject_outliers


def _find_3d_extrema(hull_verts_3d, centroid, u, v):
    """Searches for extrema in the u and v "axis" directions."""
    mags_u = np.sort(np.dot(hull_verts_3d - centroid, u))
    mags_v = np.sort(np.dot(hull_verts_3d - centroid, v))
    min_u = u * mags_u[0]
    max_u = u * mags_u[-1]
    min_v = v * mags_v[0]
    max_v = v * mags_v[-1]
    return np.array((centroid + min_u + min_v,
                     centroid + min_u + max_v,
                     centroid + max_u + max_v,
                     centroid + max_u + min_v))


def find_3d_bbox(coords_im, tangents_im, bitangents_im, region_mask) -> np.ndarray:
    """
    Finds the 3D bounding box of a planar region given a mask of a planar
    region. The bounding box is given as a 4-tuple which defines each corner
    of the bounding box in 3D.

    The algorithm is as follows:
        1. Use tangent/bitangent directions as u, v directions.
        2. Find extrema in the u, v directions which defines the corners.
    """
    hull_verts_3d = coords_im[region_mask]

    tangents = reject_outliers(tangents_im[region_mask], thres=1)
    bitangents = reject_outliers(bitangents_im[region_mask], thres=1)

    u = np.mean(tangents, axis=0)
    if linalg.norm(u) == 0:
        raise RuntimeError("Tangent is zero!")
    u /= linalg.norm(u)
    v = np.mean(bitangents, axis=0)
    if linalg.norm(v) == 0:
        raise RuntimeError("Tangent is zero!")
    v /= linalg.norm(v)

    centroid = np.mean(coords_im[region_mask], axis=0)
    return _find_3d_extrema(hull_verts_3d, centroid, u, v)


def rectify_plane(image, corners, corners_3d, scale=None):
    """
    Rectifies a region of the image defined by the bounding box. The bounding
    box is a list of corners.

    Note that we need all 4 coordinates since the corners define a
    projectively transformed rectangle.
    """
    if scale is None:
        max_len = max(linalg.norm(corners[0] - corners[1]),
                      linalg.norm(corners[2] - corners[3]),
                      linalg.norm(corners[1] - corners[2]),
                      linalg.norm(corners[0] - corners[3]))
        height = linalg.norm(corners_3d[0] - corners_3d[1])
        width = linalg.norm(corners_3d[1] - corners_3d[2])
        max_len_3d = max(height, width)
        scale = max_len / max_len_3d
        height, width = height * scale, width * scale
    else:
        height = linalg.norm(corners_3d[0] - corners_3d[1]) * scale
        width = linalg.norm(corners_3d[1] - corners_3d[2]) * scale
    reference_corners = np.array(
        ((0, 0), (height, 0), (height, width), (0, width)))
    tform = transform.ProjectiveTransform()
    tform.estimate(np.fliplr(reference_corners), np.fliplr(corners))
    rectified_image = transform.warp(
        image, inverse_map=tform, output_shape=(int(height), int(width)))
    return rectified_image
