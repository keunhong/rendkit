import os
import numpy as np

from numpy import linalg as la
from scipy import misc

from rendkit.glsl import GLSLProgram, GLSLTemplate
from vispy import app
from vispy import gloo
from vispy.gloo import gl
from vispy.util.transforms import perspective, rotate


_FACE_NAMES = {
    '+x': 0,
    '-x': 1,
    '+y': 2,
    '-y': 3,
    '+z': 4,
    '-z': 5,
}


def stack_cross(cubemap: np.ndarray):
    """
    Stacks the cubemap into an unwrapped cross.
     - Top row: +y
     - Middle row: -x, -z, +x, +z
     - Bottom row: -y
    :param cubemap: of shape 6xHxWxC
    :return: stacked cross image
    """
    height, width, channels = cubemap.shape[1:]
    result = np.zeros((height * 3, width * 4, channels))
    result[height:2*height, 2*width:3*width] = cubemap[0]
    result[height:2*height, :width] = cubemap[1]
    result[:height, width:2*width] = cubemap[2]
    result[2*height:3*height, width:2*width] = cubemap[3]
    result[height:2*height, width:2*width] = cubemap[4]
    result[height:2*height, 3*width:4*width] = cubemap[5]
    return result


def load_cubemap(path, size=(256, 256)):
    cubemap = np.zeros((6, *size, 3), dtype=np.float32)
    for fname in os.listdir(path):
        name = os.path.splitext(fname)[0]
        image = misc.imread(os.path.join(path, fname))
        image = misc.imresize(image, size).astype(np.float32) / 255.0
        cubemap[_FACE_NAMES[name]] = image
    return cubemap


class LambertPrefilterProgram(GLSLProgram):
    def __init__(self):
        super().__init__(
            GLSLTemplate.fromfile('cubemap/lambert.vert.glsl'),
            GLSLTemplate.fromfile('cubemap/lambert.frag.glsl'))

    def update_uniforms(self, program):
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return program


class LambertPrefilterProcessor:
    def __init__(self):
        self.program = LambertPrefilterProgram().compile()

    def filter(self, cubemap):
        _, height, width, n_channels = cubemap.shape
        internal_format = 'rgba32f' if n_channels == 4 else 'rgb32f'
        rendtex = gloo.Texture2D(
            (height, width, n_channels), interpolation='linear',
            wrapping='repeat', internalformat=internal_format)
        framebuffer = gloo.FrameBuffer(
            rendtex, gloo.RenderBuffer((width, height, n_channels)))
        gloo.set_viewport(0, 0, width, height)
        self.program['u_cubemap'] = gloo.TextureCubeMap(
            cubemap, internalformat=internal_format)
        results = np.zeros(cubemap.shape, dtype=np.float32)
        for i in range(6):
            self.program['u_cube_face'] = i
            with framebuffer:
                self.program.draw(gl.GL_TRIANGLE_STRIP)
                results[i] = gloo.read_pixels(out_type=np.float32, alpha=False)
        return results
