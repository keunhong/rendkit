from pathlib import Path
import numpy as np
from toolbox.images import save_hdr, load_hdr, save_image
from toolbox.logging import init_logger

logger = init_logger(__name__)


DIFF_MAP_NAME = 'map_diffuse.exr'
SPEC_MAP_NAME = 'map_specular.exr'
NORMAL_MAP_NAME = 'map_normal.exr'
BLEND_NORMAL_MAP_NAME = 'map_normal_blender.png'
ROUGH_MAP_NAME = 'map_roughness.exr'
ANISO_MAP_NAME = 'map_anisotropy.exr'


class BeckmannSVBRDF:
    """
    The Anisotropic Beckmann SVBRDF class.
    
    This uses the Blender formulation for the anisotropic Beckmann BRDF.
    The roughness and anisotropy values map to the conventional alpha_x and
    alpha_y as:
    
      if(aniso < 0.0f) {
        alpha_x = roughness/(1.0f + aniso);
        alpha_y = roughness*(1.0f + aniso);
      }
      else {
        alpha_x = roughness*(1.0f - aniso);
        alpha_y = roughness/(1.0f - aniso);
      }
    
    The Aittala BRDF has many constant factors baked into the albedos. When
    converting from the Aittala BRDF you must
        1. Multiply the diffuse albedo by PI
        2. Divide the specular albedo by 4.0*PI
    to account for the baked in constants.
    """
    @classmethod
    def from_path(cls, path):
        path = Path(path)
        logger.info("[Beckmann] Loading from {}".format(path))
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

    def save(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        logger.info("Saving Beckmann SVBRDF to {}".format(path))
        normal_map_blender = self.normal_map.copy()
        # Normalize X and Y to [0, 1] to follow blender conventions.
        normal_map_blender[:, :, :2] += 1
        normal_map_blender[:, :, :2] /= 2
        save_hdr(path / DIFF_MAP_NAME, self.diff_map)
        save_hdr(path / SPEC_MAP_NAME, self.spec_map)
        save_hdr(path / NORMAL_MAP_NAME, self.normal_map)
        save_image(path / BLEND_NORMAL_MAP_NAME, self.normal_map)
        save_hdr(path / ROUGH_MAP_NAME, self.rough_map)
        save_hdr(path / ANISO_MAP_NAME, self.aniso_map)
