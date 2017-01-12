import numpy as np
from vispy import gloo


class Light:
    pass


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
    def __init__(self, array: np.ndarray, scale=1.0):
        if array.dtype != np.float32:
            array = array.astype(dtype=np.float32) / 255.0
        if len(array.shape) < 3:
            array = np.repeat(array[:, :, None], 3, axis=2)
        array = array[:, :, :3]
        self.array = array * scale
        self.texture = gloo.Texture2D(self.array,
                                      interpolation='linear',
                                      internalformat='rgb32f')

    @property
    def size(self):
        return self.array.shape

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value
        self.texture.set_data(self.array)
