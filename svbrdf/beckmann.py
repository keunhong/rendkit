from pathlib import Path
import numpy as np
from toolbox.images import save_hdr, load_hdr

DIFF_MAP_NAME = 'diff_map.exr'
SPEC_MAP_NAME = 'spec_map.exr'
NORMAL_MAP_NAME = 'normal_map.exr'
ROUGH_MAP_NAME = 'rough_map.exr'
ANISO_MAP_NAME = 'aniso_map.exr'


class BeckmannSVBRDF:
    @classmethod
    def from_path(cls, path):
        return BeckmannSVBRDF(
            diff_map=load_hdr(path / DIFF_MAP_NAME),
            spec_map=load_hdr(path / SPEC_MAP_NAME),
            normal_map=load_hdr(path / NORMAL_MAP_NAME),
            rough_map=load_hdr(path / ROUGH_MAP_NAME),
            aniso_map=load_hdr(path / ANISO_MAP_NAME))

    def __init__(self, diff_map, spec_map, normal_map, rough_map, aniso_map):
        self.diff_map = diff_map.astype(np.float32)
        self.spec_map = spec_map.astype(np.float32)
        self.normal_map = normal_map.astype(np.float32)
        self.rough_map = rough_map.astype(np.float32)
        self.aniso_map = aniso_map.astype(np.float32)

    def save(self, path, fmt='exr'):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        save_hdr(path / 'diff_map.{}'.format(fmt), self.diff_map, fmt=fmt)
        save_hdr(path / 'spec_map.{}'.format(fmt), self.spec_map, fmt=fmt)
        save_hdr(path / 'normal_map.{}'.format(fmt), self.normal_map, fmt=fmt)
        save_hdr(path / 'rough_map.{}'.format(fmt), self.rough_map, fmt=fmt)
        save_hdr(path / 'aniso_map.{}'.format(fmt), self.aniso_map, fmt=fmt)
