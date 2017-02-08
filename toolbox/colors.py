from copy import copy

import numpy as np
from scipy.ndimage import maximum_filter
from skimage.color import rgb2lab, lab2rgb
from sklearn.mixture import GaussianMixture
from toolbox.images import apply_mask

from rendkit.shortcuts import svbrdf_plane_renderer


def _gmm_transfer_color(reference_gmm, target_gmm, target_lab):
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
        recolored_lab[target_mask] = target_vals * reference_std + reference_mean

    return recolored_lab


def _lab_color_gmm(lab, mask, n_components):
    gmm = GaussianMixture(n_components=n_components)
    print(np.sum(mask))
    data = lab[mask].reshape(-1, 3)
    gmm.fit(data)
    return gmm


def _percentile_mask(im, lo=0, hi=95):
    mean_im = im.mean(axis=2)
    percentile_lo = np.percentile(mean_im, lo)
    percentile_hi = np.percentile(mean_im, hi)
    return (mean_im > percentile_lo) & (mean_im < percentile_hi)


def match_color_gmm(reference_gmm, target_im, target_mask=None,
                    out_colors=None):
    if target_mask is None:
        target_mask = np.ones(target_im.shape[:2], dtype=bool)

    n_components = reference_gmm.n_components

    print("Computing target GMM")
    target_lab = rgb2lab(np.clip(target_im, 0, 1))
    target_gmm = _lab_color_gmm(target_lab[:, :, :],
                                target_mask[:, :],
                                n_components)
    if out_colors is not None:
        print("Setting out_colors!")
        print(reference_gmm.means_, target_gmm.means_)
        out_colors[:50,:50, :] = reference_gmm.means_[0]
        out_colors[:50,50:, :] = reference_gmm.means_[1]
        out_colors[50:,:50, :] = target_gmm.means_[0]
        out_colors[50:,50:, :] = target_gmm.means_[1]
        out_colors[:, :, :] = lab2rgb(out_colors) ** (1/2.2)

    recolored_lab = _gmm_transfer_color(reference_gmm, target_gmm, target_lab)
    return lab2rgb(recolored_lab)

from scipy import misc
def match_color(reference_im, target_im,
                reference_mask=None,
                target_mask=None,
                n_components=2, out_colors=None):
    if reference_mask is None:
        reference_mask = np.ones(reference_im.shape[:2], dtype=bool)
    print(reference_im.min(), reference_im.max(), target_im.min(), target_im.max())

    print('Computing reference GMM')
    reference_lab = rgb2lab(reference_im)
    reference_gmm = _lab_color_gmm(reference_lab, reference_mask, n_components)

    return match_color_gmm(reference_gmm, target_im, target_mask, out_colors=out_colors)


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


def svbrdf_match_color(reference_im, svbrdf, mask=None, out_colors=None):
    radmap = np.clip(np.random.normal(1.0, 2.0, (20, 20)), 0, None)
    radmap_jsd = dict(type='inline', array=radmap)
    with svbrdf_plane_renderer(
            svbrdf, mode='light_map', radiance_map=radmap_jsd) as renderer:
        light_map = renderer.render_to_image()
    if mask is None:
        mask = np.ones(reference_im.shape[:2], dtype=bool)
    target_map = np.clip(svbrdf.diffuse_map * light_map.mean(), 0, 1)
    reference_mask = mask# & _percentile_mask(reference_im, lo=0, hi=95)
    target_mask = _percentile_mask(target_map, lo=0, hi=100)
    # target_mask = np.ones(target_map.shape[:2], dtype=bool)
    recolored_map = match_color(reference_im, target_map,
                                reference_mask=reference_mask,
                                target_mask=target_mask,
                                out_colors=out_colors)
    # Undo gamma correction and lighting effects.
    recolored_map = recolored_map / light_map.mean()
    recolored_svbrdf = copy(svbrdf)
    recolored_svbrdf.diffuse_map = recolored_map.astype(dtype=np.float32)

    return recolored_svbrdf


def svbrdf_match_color_gmm(reference_gmm, svbrdf):
    light_map = svbrdf_plane_renderer(svbrdf,
                                      mode='light_map').render_to_image()
    target_map = (svbrdf.diffuse_map * light_map.mean())
    target_mask = _percentile_mask(target_map, lo=0, hi=100)
    recolored_map = match_color_gmm(reference_gmm, target_map,
                                    target_mask=target_mask)
    # Undo gamma correction and lighting effects.
    recolored_map = recolored_map / light_map.mean()
    recolored_svbrdf = copy(svbrdf)
    recolored_svbrdf.diffuse_map = recolored_map.astype(dtype=np.float32)

    return recolored_svbrdf
