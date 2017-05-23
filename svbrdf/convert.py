import math
import argparse
import itertools
import torch
import numpy as np
from scipy.interpolate import griddata
from tqdm import trange

from svbrdf.beckmann import BeckmannSVBRDF
from toolbox.logging import init_logger
from svbrdf.aittala import AittalaSVBRDF

logger = init_logger(__name__)


def aittala_ndf(H, S, alpha=2):
    S = S.view(1, 2, 2).expand(H.size(0), 2, 2)
    h = H.view(-1, 2, 1)
    hT = H.view(-1, 1, 2)
    e = (torch.bmm(hT, torch.bmm(S, h)))
    e = e.abs() ** alpha
    return torch.exp(-e)


def angles_to_half(angles, tangent=False):
    H = torch.zeros(3, len(angles)).cuda()
    phi = angles[:, 0]
    theta = angles[:, 1]
    H[0] = torch.cos(theta) * torch.sin(phi)
    H[1] = torch.sin(theta) * torch.sin(phi)
    H[2] = torch.cos(phi)
    return H


def beckmann_aniso_ndf(H, alpha_x, alpha_y):
    hx = H[0].view(-1, 1)
    hy = H[1].view(-1, 1)
    hz = H[2].view(-1, 1)

    if alpha_x == 0 or alpha_y == 0:
        return None

    slope_x = (-hx / (hz * alpha_x))
    slope_y = (-hy / (hz * alpha_y))
    cosThetaM = hz
    cosThetaM2 = cosThetaM * cosThetaM
    cosThetaM4 = cosThetaM2 * cosThetaM2
    cosThetaM4 = cosThetaM4.expand(*slope_x.size())
    D = torch.exp(-slope_x * slope_x - slope_y * slope_y) / cosThetaM4
    return D


def fit_beckmann(D_aittala, H, ones):
    ax_best = 0.5
    ay_best = 0.5
    ax_lo, ax_hi = 0, 1
    ay_lo, ay_hi = 0, 1
    best_err = float('inf')

    win_size = 3
    for branch in range(7):
        ax_cand = np.linspace(ax_lo, ax_hi, win_size)
        ay_cand = np.linspace(ay_lo, ay_hi, win_size)
        param_cand = itertools.product(ax_cand, ay_cand)
        for ax, ay in param_cand:
            D = beckmann_aniso_ndf(H, ax, ay)
            if D is None:
                D = ones
            err = (D - D_aittala).abs()
            err = err.sum()
            if err < best_err:
                ax_best = ax
                ay_best = ay
                best_err = err
        ax_win = (ax_hi - ax_lo) / (win_size)
        ay_win = (ax_hi - ax_lo) / (win_size)
        ax_lo, ax_hi = max(0, ax_best - ax_win), min(1, ax_best + ax_win)
        ay_lo, ay_hi = max(0, ay_best - ay_win), min(1, ay_best + ay_win)
    return ax_best, ay_best, best_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--aittala', dest='aittala_path', type=str,
                        required=True)
    parser.add_argument('-o', '--out', dest='out_path', type=str,
                        required=True)
    args = parser.parse_args()

    svbrdf = AittalaSVBRDF(args.aittala_path)
    crop_size = (200, 200)

    crop_spec_shape = svbrdf.spec_shape_map[:crop_size[0], :crop_size[1]].reshape((-1, 3))
    crop_spec_shape = crop_spec_shape[:, [0, 1, 1, 2]].reshape((-1, 2, 2))
    crop_spec_shape = torch.from_numpy(crop_spec_shape).cuda()

    n_phi = 500
    n_theta_max = 800
    logger.info("Generating sample angles..")
    phis = np.linspace(0.0, math.pi / 2, n_phi)
    angles = []
    for phi in phis:
        n_theta = int(round(n_theta_max * math.sin(phi) + 1))
        thetas = np.linspace(0, math.pi * 2, n_theta)
        for theta in thetas:
            angles.append((phi, theta))
    H = angles_to_half(torch.FloatTensor(angles))
    H_tan = H[:2, :] / H[2].contiguous().view(1, -1).expand(*H[:2, :].size())
    H_tan = H_tan.t().contiguous()

    logger.info("Fitting Beckmann BRDF...")
    ones = torch.ones(H.size(1)).cuda()
    crop_rough_map = np.zeros(crop_size)
    crop_aniso_map = np.zeros(crop_size)
    for i in trange(crop_spec_shape.size(0)):
        row, col = i // crop_size[0], i % crop_size[1]
        D_aittala = aittala_ndf(H_tan, crop_spec_shape[i], alpha=svbrdf.alpha)
        ax, ay, err = fit_beckmann(D_aittala, H, ones)
        if ay > ax:
            aniso = 1 - math.sqrt(ax / ay)
            roughness = ay * (1 - aniso)
        else:
            aniso = math.sqrt(ay / ax) - 1
            roughness = ax * (1 + aniso)
        crop_aniso_map[row, col] = aniso
        crop_rough_map[row, col] = roughness

    logger.info("Interpolating sample fit to full texture...")
    crop_spec_shape = svbrdf.spec_shape_map[:crop_size[0],
                                            :crop_size[1]].reshape((-1, 3))
    rough_map = griddata(crop_spec_shape, crop_rough_map.flatten(),
                         svbrdf.spec_shape_map.reshape((-1, 3)),
                         method='nearest')
    rough_map = rough_map.reshape(svbrdf.spec_shape_map.shape[:2])
    aniso_map = griddata(crop_spec_shape, crop_aniso_map.flatten(),
                         svbrdf.spec_shape_map.reshape((-1, 3)),
                         method='nearest')
    aniso_map = aniso_map.reshape(svbrdf.spec_shape_map.shape[:2])

    logger.info("Saving...")
    bsvbrdf = BeckmannSVBRDF(svbrdf.diffuse_map,
                             svbrdf.specular_map,
                             svbrdf.normal_map,
                             rough_map,
                             aniso_map)
    bsvbrdf.save(args.out_path)


if __name__ == '__main__':
    main()
