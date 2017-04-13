import logging
from copy import copy

import numpy as np
from scipy.ndimage import maximum_filter
from skimage.color import rgb2lab, lab2rgb
from sklearn.mixture import GaussianMixture

from rendkit.shortcuts import svbrdf_plane_renderer
from toolbox.images import apply_mask


logger = logging.getLogger(__name__)


def _gmm_coerce_distribution(reference_gmm, target_gmm, target_lab):
    # Predict which color "segment" each pixel is in.
    target_pred = target_gmm.predict(target_lab.reshape(-1, 3)).reshape(
        target_lab.shape[:2])

    reference_lum = reference_gmm.means_[:, 0]
    reference_component_ordering = np.argsort(reference_lum)
    target_lum = target_gmm.means_[:, 0]
    target_component_ordering = np.argsort(target_lum)
    recolored_lab = np.zeros(target_lab.shape)
    for i in range(len(reference_gmm.means_)):
        reference_idx = reference_component_ordering[i]
        target_idx = target_component_ordering[i]

        reference_mean = reference_gmm.means_[reference_idx]
        reference_std = np.sqrt(np.diag(
            reference_gmm.covariances_[reference_idx, :]))
        target_mean = target_gmm.means_[target_idx]
        target_std = np.sqrt(np.diag(target_gmm.covariances_[target_idx, :]))

        target_mask = target_pred == target_idx
        target_vals = target_lab[target_mask]
        target_vals = (target_vals - target_mean) / target_std
        new_std = 0.5 * target_std + 0.5 * reference_std
        recolored_lab[target_mask] = target_vals * new_std + reference_mean

    return recolored_lab


def _compute_image_gmm(image_lab, mask, n_components):
    gmm = GaussianMixture(n_components=n_components)
    data = image_lab[mask].reshape(-1, 3)
    gmm.fit(data)
    return gmm


def _percentile_mask(im, lo=0, hi=95, mask=None):
    if mask is None:
        mask = np.ones(im.shape[:2], dtype=bool)
    mean_im = im.mean(axis=2)
    percentile_lo = np.percentile(mean_im[mask], lo)
    percentile_hi = np.percentile(mean_im[mask], hi)
    return (mean_im > percentile_lo) & (mean_im < percentile_hi)


def visualize_color(colors, size=50):
    n_colors = colors.shape[0]
    vis = np.zeros((size, n_colors*size, 3))
    for i in range(n_colors):
        vis[:, i*size:i*size+size] = colors[i]
    return vis


def match_color(reference_im,
                target_im,
                reference_mask=None,
                target_mask=None,
                n_components=2):
    if reference_mask is None:
        reference_mask = np.ones(reference_im.shape[:2], dtype=bool)
    if target_mask is None:
        target_mask = np.ones(target_im.shape[:2], dtype=bool)
    logger.info("Matching color (n_components={}, ref_size={}, tar_size={})"
                .format(n_components, np.sum(reference_mask), np.sum(target_mask)))
    reference_lab = rgb2lab(np.clip(reference_im, 0, 1))
    reference_gmm = _compute_image_gmm(reference_lab, reference_mask, n_components)

    target_lab = rgb2lab(np.clip(target_im, 0, 1))
    target_gmm = _compute_image_gmm(target_lab, target_mask, n_components)

    result_rgb = lab2rgb(
        _gmm_coerce_distribution(reference_gmm, target_gmm, target_lab))
    reference_means = lab2rgb(
        reference_gmm.means_.reshape((1, n_components, 3)))[0]
    target_means = lab2rgb(
        target_gmm.means_.reshape((1, n_components, 3)))[0]
    return result_rgb, reference_means, target_means


def compute_color_gmm_iter(image, mask, n_iters=10):
    masked_image = apply_mask(image, mask)
    image_lab = rgb2lab(masked_image)
    gmm = GaussianMixture(n_components=2)

    current_mask = mask.copy()
    for i in range(n_iters):
        gmm.fit(image_lab[current_mask].reshape(-1, 3))
        print("Iteration {}, means={}".format(i, gmm.means_))
        rm1 = (np.abs(image_lab - gmm.means_[0]) < 1.0 * np.sqrt(
            np.diag(gmm.covariances_[0])))
        rm2 = (np.abs(image_lab - gmm.means_[1]) < 1.0 * np.sqrt(
            np.diag(gmm.covariances_[1])))
        rm1 = rm1[:, :, 0]
        rm2 = rm2[:, :, 0]
        current_mask = maximum_filter(mask & (rm1 | rm2), size=5)

    return gmm, current_mask


def svbrdf_match_color(reference_im, svbrdf, mask=None, radmap=None):
    if radmap is None:
        radmap = np.clip(np.random.normal(1.0, 2.0, (20, 20)), 0, None)
    radmap_jsd = dict(type='inline', array=radmap)
    with svbrdf_plane_renderer(
            svbrdf, mode='light_map', radmap=radmap_jsd, gamma=None) as renderer:
        light_map = renderer.render_to_image()
        target_map = np.clip(svbrdf.diffuse_map * light_map, 0, 1)
    if mask is None:
        mask = np.ones(reference_im.shape[:2], dtype=bool)
    reference_mask = _percentile_mask(reference_im, lo=1, hi=95, mask=mask)
    recolored_map = match_color(reference_im, target_map,
                                reference_mask=reference_mask)
    # Undo gamma correction and lighting effects.
    recolored_map = recolored_map / light_map
    recolored_svbrdf = copy(svbrdf)
    recolored_svbrdf.diffuse_map = recolored_map.astype(dtype=np.float32)

    return recolored_svbrdf


def normalize_lab(lab_values):
    return (lab_values - (50, 0, 0)) / (50, 128, 128)


def denormalize_lab(norm_lab_values):
    return np.array(norm_lab_values) * (50, 128, 128) + (50, 0, 0)


def hist_match(source, template, source_mask=None, template_mask=None):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    if source_mask is None:
        source_mask = np.ones(source.shape[:2], dtype=bool)
    if template_mask is None:
        template_mask = np.ones(template.shape[:2], dtype=bool)

    result = source.copy()
    source = source[source_mask].ravel()
    template = template[template_mask].ravel()

    if len(source) < 1 or len(template) < 0:
        logger.warning("Source and target are empty.")
        return np.zeros(source.shape)

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    result[source_mask] = interp_t_values[bin_idx]
    result[~source_mask] = np.median(result[source_mask], axis=-1)
    return result
