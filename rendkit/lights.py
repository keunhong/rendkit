import logging
import numpy as np
from vispy import gloo
from rendkit.cubemap import prefilter_irradiance


logger = logging.getLogger(__name__)


class Light:
    pass


class AmbientLight(Light):
    type = 2

    def __init__(self, position, intensity, color=(1.0, 1.0, 1.0)):
        self.intensity = intensity
        self.color = color


class PointLight(Light):
    type = 0

    def __init__(self, position, intensity, color=(1.0, 1.0, 1.0)):
        self.position = position
        self.intensity = intensity
        self.color = color


class DirectionalLight(Light):
    type = 1

    def __init__(self, direction, intensity, color=(1.0, 1.0, 1.0)):
        self.position = direction
        self.intensity = intensity
        self.color = color


class RadianceMap(Light):
    def __init__(self, cube_faces: np.ndarray, scale=1.0):
        if cube_faces.shape[0] != 6:
            raise RuntimeError('Cubemap must have exactly 6 faces.')
        self.cube_faces = cube_faces * scale
        logger.info("Prefiltering irradiance map.")
        self.irradiance_map = prefilter_irradiance(self.cube_faces)
        self.irradiance_map_tex = gloo.TextureCubeMap(
            self.irradiance_map, interpolation='linear', internalformat='rgb32f')
        self.radiance_map_tex = gloo.TextureCubeMap(
            self.cube_faces, interpolation='linear', internalformat='rgb32f')

    @property
    def size(self):
        return self.cube_faces.shape[1:3]
