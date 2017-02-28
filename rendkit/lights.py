import logging
import numpy as np

from rendkit.renderers import DummyRenderer
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
        if gloo.get_current_canvas() is None:
            raise RuntimeError("OpenGL context is required for prefiltering. "
                               "Please initialize with DummyRenderer().")
        if cube_faces.shape[0] != 6:
            raise RuntimeError('Cubemap must have exactly 6 faces.')
        face_height, face_width = cube_faces.shape[1:3]
        self.radiance_faces = cube_faces * scale
        logger.info("Prefiltering irradiance map.")
        with DummyRenderer(size=(face_width, face_height)):
            self.irradiance_faces = prefilter_irradiance(self.radiance_faces)

    @property
    def radiance_faces(self):
        return self._radiance_faces

    @radiance_faces.setter
    def radiance_faces(self, radiance_faces):
        self._radiance_faces = radiance_faces
        self.radiance_map_tex = gloo.TextureCubeMap(
            self.radiance_faces, interpolation='linear', internalformat='rgb32f')

    @property
    def irradiance_faces(self):
        return self._irradiance_faces

    @irradiance_faces.setter
    def irradiance_faces(self, irradiance_faces):
        self._irradiance_faces = irradiance_faces
        self.irradiance_map_tex = gloo.TextureCubeMap(
            self.irradiance_faces, interpolation='linear', internalformat='rgb32f')

    @property
    def size(self):
        return self.radiance_faces.shape[1:3]
