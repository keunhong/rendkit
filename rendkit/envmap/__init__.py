import numpy as np

from rendkit.envmap.prefilter import prefilter_irradiance
from rendkit.envmap.conversion import cubemap_to_dual_paraboloid
from rendkit.lights import logger
from vispy import gloo


class EnvironmentMap():
    def __init__(self, cube_faces: np.ndarray, scale=1.0):
        if cube_faces.shape[0] != 6:
            raise RuntimeError('Cubemap must have exactly 6 faces.')
        self._radiance_faces = None
        self._radiance_tex = None
        self._irradiance_faces = None
        self._irradiance_tex = None
        self._radiance_upper_tex = None
        self._radiance_lower_tex = None

        self.radiance_faces = cube_faces * scale
        logger.info("Prefiltering irradiance map.")
        self.irradiance_faces = prefilter_irradiance(self.radiance_faces)

    @property
    def radiance_faces(self):
        return self._radiance_faces

    @radiance_faces.setter
    def radiance_faces(self, radiance_faces):
        self._radiance_faces = radiance_faces
        upper_map, lower_map = cubemap_to_dual_paraboloid(self.radiance_faces)
        self._radiance_upper_tex = gloo.Texture2D(
            upper_map,
            interpolation='linear_mipmap_linear',
            internalformat='rgb32f',
            mipmap_levels=8)
        self._radiance_lower_tex = gloo.Texture2D(
            lower_map,
            interpolation='linear_mipmap_linear',
            internalformat='rgb32f',
            mipmap_levels=8)

    @property
    def irradiance_faces(self):
        return self._irradiance_faces

    @irradiance_faces.setter
    def irradiance_faces(self, irradiance_faces):
        print('asdf')
        self._irradiance_faces = irradiance_faces
        self._irradiance_tex = gloo.TextureCubeMap(
            self._irradiance_faces,
            interpolation='linear',
            internalformat='rgb32f')

    @property
    def radiance_upper_tex(self):
        return self._radiance_upper_tex

    @property
    def radiance_lower_tex(self):
        return self._radiance_lower_tex

    @property
    def irradiance_tex(self):
        return self._irradiance_tex

    @property
    def size(self):
        return (self.radiance_faces.shape[2], self.radiance_faces.shape[1])

    def reset(self):
        self.irradiance_faces = self._irradiance_faces
        self.radiance_faces = self._radiance_faces
