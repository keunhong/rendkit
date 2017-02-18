import os
import numpy as np

from scipy import misc

from rendkit.glsl import GLSLProgram, GLSLTemplate
from vispy import gloo
from vispy.gloo import gl


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


def stack_cross(cube_faces: np.ndarray):
    """
    Stacks the cubemap into an unwrapped cross.
     - Top row: +y
     - Middle row: -x, -z, +x, +z
     - Bottom row: -y
    :param cube_faces: of shape 6xHxWxC
    :return: stacked cross image
    """
    _, height, width, n_channels = cube_faces.shape
    result = np.zeros((height * 3, width * 4, n_channels))
    result[height:2*height, 2*width:3*width] = cube_faces[0]
    result[height:2*height, :width] = cube_faces[1]
    result[:height, width:2*width] = cube_faces[2]
    result[2*height:3*height, width:2*width] = cube_faces[3]
    result[height:2*height, width:2*width] = cube_faces[4]
    result[height:2*height, 3*width:4*width] = cube_faces[5]
    return result


def unstack_cross(cross):
    assert cross.shape[0] % 3 == 0
    assert cross.shape[1] % 4 == 0
    height, width = cross.shape[0] // 3, cross.shape[1] // 4
    n_channels = cross.shape[2]
    faces = np.zeros((6, height, width, n_channels), dtype=np.float32)
    faces[0] = cross[height:2 * height, 2 * width:3 * width]
    faces[1] = cross[height:2 * height, :width]
    faces[2] = cross[:height, width:2 * width]
    faces[3] = cross[2 * height:3 * height, width:2 * width]
    faces[4] = cross[height:2 * height, width:2 * width]
    faces[5] = cross[height:2 * height, 3 * width:4 * width]
    return faces


def load_cube_faces(path, size=(256, 256)):
    cubemap = np.zeros((6, *size, 3), dtype=np.float32)
    for fname in os.listdir(path):
        name = os.path.splitext(fname)[0]
        image = misc.imread(os.path.join(path, fname))
        image = misc.imresize(image, size).astype(np.float32) / 255.0
        cubemap[_FACE_NAMES[name]] = image
    return cubemap


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
