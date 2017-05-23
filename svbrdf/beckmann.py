from pathlib import Path
import numpy as np
import cv2

from rendkit import pfm


class BeckmannSVBRDF:
    def __init__(self, diff_map, spec_map, normal_map, rough_map, aniso_map):
        self.diff_map = diff_map.astype(np.float32)
        self.spec_map = spec_map.astype(np.float32)
        self.normal_map = normal_map.astype(np.float32)
        self.rough_map = rough_map.astype(np.float32)
        self.aniso_map = aniso_map.astype(np.float32)

    def save(self, path, fmt='exr'):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        if fmt == 'exr':
            cv2.imwrite(str(path / 'diff_map.exr'), self.diff_map[:, :, [2, 1, 0]])
            cv2.imwrite(str(path / 'spec_map.exr'), self.spec_map[:, :, [2, 1, 0]])
            cv2.imwrite(str(path / 'normal_map.exr'), self.normal_map[:, :, [2, 1, 0]])
            cv2.imwrite(str(path / 'rough_map.exr'), self.rough_map)
            cv2.imwrite(str(path / 'aniso_map.exr'), self.aniso_map)
        elif fmt == 'pfm':
            pfm.pfm_write(path / 'diff_map.exr', self.diff_map)
            pfm.pfm_write(path / 'spec_map.exr', self.spec_map)
            pfm.pfm_write(path / 'normal_map.exr', self.normal_map)
            pfm.pfm_write(path / 'rough_map.exr', self.rough_map)
            pfm.pfm_write(path / 'aniso_map.exr', self.aniso_map)
        else:
            raise RuntimeError("Unknown format {}".format(fmt))
