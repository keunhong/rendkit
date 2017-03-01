import os
import numpy as np
from functools import partial

from scipy import misc
from scipy.misc import imread

from rendkit import pfm
from rendkit.glsl import GLSLProgram, GLSLTemplate
from rendkit.lights import logger
from rendkit.renderers import ContextProvider
from vispy import gloo
from vispy.gloo import gl
from toolbox.images import resize


_FACE_NAMES = {
    '+x': 0,
    '-x': 1,
    '+y': 2,
    '-y': 3,
    '+z': 4,
    '-z': 5,
}


class LambertPrefilterProgram(GLSLProgram):
    def __init__(self):
        super().__init__(
            GLSLTemplate.fromfile('cubemap/lambert.vert.glsl'),
            GLSLTemplate.fromfile('cubemap/lambert.frag.glsl'))

    def update_uniforms(self, program):
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return program


def _set_grid(grid: np.ndarray, height, width, u, v, value):
    grid[u*height:(u+1)*height, v*width:(v+1)*width] = value


def _get_grid(grid, height, width, u, v):
    return grid[u*height:(u+1)*height, v*width:(v+1)*width]


def stack_cross(cube_faces: np.ndarray, format='vertical'):
    _, height, width, n_channels = cube_faces.shape
    if format == 'vertical':
        result = np.zeros((height * 4, width * 3, n_channels))
        gridf = partial(_set_grid, result, height, width)
        gridf(0, 1, cube_faces[_FACE_NAMES['+y']])
        gridf(1, 0, cube_faces[_FACE_NAMES['-x']])
        gridf(1, 1, cube_faces[_FACE_NAMES['+z']])
        gridf(1, 2, cube_faces[_FACE_NAMES['+x']])
        gridf(2, 1, cube_faces[_FACE_NAMES['-y']])
        gridf(3, 1, np.fliplr(np.flipud(cube_faces[_FACE_NAMES['-z']])))
    elif format == 'horizontal':
        result = np.zeros((height * 3, width * 4, n_channels))
        gridf = partial(_set_grid, result, height, width)
        gridf(1, 2, cube_faces[_FACE_NAMES['+x']])
        gridf(1, 0, cube_faces[_FACE_NAMES['-x']])
        gridf(0, 1, cube_faces[_FACE_NAMES['+y']])
        gridf(2, 1, cube_faces[_FACE_NAMES['-y']])
        gridf(1, 1, cube_faces[_FACE_NAMES['+z']])
        gridf(1, 3, cube_faces[_FACE_NAMES['-z']])
    else:
        raise RuntimeError("Unknown format {}".format(format))
    return result


def unstack_cross(cross):
    if cross.shape[0] % 3 == 0 and cross.shape[1] % 4 == 0:
        format = 'horizontal'
        height, width = cross.shape[0] // 3, cross.shape[1] // 4
    elif cross.shape[0] % 4 == 0 and cross.shape[1] % 3 == 0:
        format = 'vertical'
        height, width = cross.shape[0] // 4, cross.shape[1] // 3
    else:
        raise RuntimeError("Unknown cross format.")

    n_channels = cross.shape[2]
    faces = np.zeros((6, height, width, n_channels), dtype=np.float32)
    gridf = partial(_get_grid, cross, height, width)

    if format == 'vertical':
        faces[0] = gridf(1, 2)
        faces[1] = gridf(1, 0)
        faces[2] = gridf(0, 1)
        faces[3] = gridf(2, 1)
        faces[4] = gridf(1, 1)
        faces[5] = np.flipud(np.fliplr(gridf(3, 1)))
    elif format == 'horizontal':
        faces[0] = gridf(1, 2)
        faces[1] = gridf(1, 0)
        faces[2] = gridf(0, 1)
        faces[3] = gridf(2, 1)
        faces[4] = gridf(1, 1)
        faces[5] = gridf(1, 3)
    return faces


def load_cubemap(path, size=(256, 256)):
    ext = os.path.splitext(path)[1]
    shape = (*size, 3)
    cube_faces = np.zeros((6, *shape), dtype=np.float32)
    if os.path.isdir(path):
        for fname in os.listdir(path):
            name = os.path.splitext(fname)[0]
            image = misc.imread(os.path.join(path, fname))
            image = misc.imresize(image, size).astype(np.float32) / 255.0
            cube_faces[_FACE_NAMES[name]] = image
    elif ext == '.pfm':
        array = np.flipud(pfm.load_pfm_texture(path))
        for i, face in enumerate(unstack_cross(array)):
            cube_faces[i] = resize(face, size)[:, :, :3]
    elif ext == '.jpg' or ext == '.png' or ext == '.tiff':
        array = imread(path)
        for i, face in enumerate(unstack_cross(array)):
            cube_faces[i] = resize(face, size)[:, :, :3]
    else:
        raise RuntimeError("Unknown cube map format.")
    return cube_faces


def prefilter_irradiance(cube_faces):
    program = LambertPrefilterProgram().compile()
    _, height, width, n_channels = cube_faces.shape
    internal_format = 'rgba32f' if n_channels == 4 else 'rgb32f'
    rendtex = gloo.Texture2D(
        (height, width, n_channels), interpolation='linear',
        wrapping='repeat', internalformat=internal_format)
    framebuffer = gloo.FrameBuffer(
        rendtex, gloo.RenderBuffer((width, height, n_channels)))
    gloo.set_viewport(0, 0, width, height)
    program['u_cubemap'] = gloo.TextureCubeMap(
        cube_faces, internalformat=internal_format)
    results = np.zeros(cube_faces.shape, dtype=np.float32)
    for i in range(6):
        program['u_cube_face'] = i
        with framebuffer:
            program.draw(gl.GL_TRIANGLE_STRIP)
            results[i] = gloo.read_pixels(out_type=np.float32, alpha=False)
    return results


class RadianceMap():
    def __init__(self, cube_faces: np.ndarray, scale=1.0):
        if cube_faces.shape[0] != 6:
            raise RuntimeError('Cubemap must have exactly 6 faces.')
        face_height, face_width = cube_faces.shape[1:3]
        self._radiance_faces = None
        self._radiance_tex = None
        self._irradiance_faces = None
        self._irradiance_tex = None

        self.radiance_faces = cube_faces * scale
        logger.info("Prefiltering irradiance map.")
        with ContextProvider((face_width, face_height)):
            self.irradiance_faces = prefilter_irradiance(self.radiance_faces)

    @property
    def radiance_faces(self):
        return self._radiance_faces

    @radiance_faces.setter
    def radiance_faces(self, radiance_faces):
        self._radiance_faces = radiance_faces
        self._radiance_tex = gloo.TextureCubeMap(
            self.radiance_faces,
            interpolation='linear',
            internalformat='rgb32f')

    @property
    def irradiance_faces(self):
        return self._irradiance_faces

    @irradiance_faces.setter
    def irradiance_faces(self, irradiance_faces):
        self._irradiance_faces = irradiance_faces
        self._irradiance_tex = gloo.TextureCubeMap(
            self._irradiance_faces,
            interpolation='linear',
            internalformat='rgb32f')

    @property
    def radiance_tex(self):
        return self._radiance_tex

    @property
    def irradiance_tex(self):
        return self._irradiance_tex

    @property
    def size(self):
        return self.radiance_faces.shape[1:3]
