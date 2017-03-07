import os
import numpy as np
from functools import partial

from scipy import misc
from scipy.misc import imread
from skimage import morphology as morph
from skimage.measure import regionprops

from rendkit import pfm
from rendkit import vector_utils
from rendkit.glsl import GLSLProgram, GLSLTemplate
from rendkit.lights import logger
from rendkit.renderers import ContextProvider
from vispy import gloo
from vispy.gloo import gl
from toolbox.images import resize, rgb2gray

_FACE_NAMES = {
    '+x': 0,
    '-x': 1,
    '+y': 2,
    '-y': 3,
    '+z': 4,
    '-z': 5,
}


def cubemap_uv_to_xyz(index, u, v):
    uc = 2.0 * u - 1.0
    vc = 2.0 * v - 1.0
    return vector_utils.normalized({
        0: (1.0, vc, -uc),
        1: (-1.0,vc, uc),
        2: (uc,1.0, -vc),
        3: (uc,-1.0, vc),
        4: (uc, vc, 1.0),
        5: (-uc, vc, -1.0),
    }[index])


def find_shadow_sources(cubemap, num_shadows=8):
    thres = np.percentile(cubemap, 90)
    shadow_source_mask = rgb2gray(stack_cross(cubemap)) >= thres
    shadow_source_mask = morph.remove_small_objects(shadow_source_mask, min_size=128)
    shadow_source_labels, num_labels = morph.label(shadow_source_mask, return_num=True)
    shadow_source_labels = unstack_cross(shadow_source_labels).astype(dtype=int)
    shadow_positions = []
    height, width = cubemap.shape[1:3]
    for i in range(6):
        face_props = regionprops(shadow_source_labels[i])
        for p in face_props:
            shadow_pos = cubemap_uv_to_xyz(
                i, p.centroid[0]/ height, p.centroid[1] / width)
            if shadow_pos[1] > 0:
                intensity = cubemap[i][shadow_source_labels[i] == p.label].mean()
                shadow_positions.append((shadow_pos, p.area, intensity))
    shadow_positions.sort(key=lambda v: -v[1] * v[2])
    print(shadow_positions[:num_shadows])
    return [p[0] for p in shadow_positions[:num_shadows]]


class LambertPrefilterProgram(GLSLProgram):
    def __init__(self):
        super().__init__(
            GLSLTemplate.fromfile('cubemap/lambert.vert.glsl'),
            GLSLTemplate.fromfile('cubemap/lambert.frag.glsl'))

    def update_uniforms(self, program):
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return program


def prefilter_irradiance(cube_faces):
    program = LambertPrefilterProgram().compile()
    _, height, width, n_channels = cube_faces.shape
    internal_format = 'rgba32f' if n_channels == 4 else 'rgb32f'

    with ContextProvider((height, width)):
        rendtex = gloo.Texture2D(
            (height, width, n_channels), interpolation='linear',
            wrapping='repeat', internalformat=internal_format)
        framebuffer = gloo.FrameBuffer(
            rendtex, gloo.RenderBuffer((width, height, n_channels)))
        gloo.set_viewport(0, 0, width, height)
        program['u_radiance_map'] = gloo.TextureCubeMap(
            cube_faces, internalformat=internal_format, mipmap_levels=8)
        program['u_cubemap_size'] = (width, height)
        results = np.zeros(cube_faces.shape, dtype=np.float32)
        for i in range(6):
            program['u_cube_face'] = i
            with framebuffer:
                program.draw(gl.GL_TRIANGLE_STRIP)
                results[i] = gloo.read_pixels(out_type=np.float32, format='rgb')
    return results


class CubemapToDualParaboloidProgram(GLSLProgram):
    def __init__(self):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('cubemap/cubemap_to_dual_paraboloid.frag.glsl'))

    def update_uniforms(self, program):
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['a_uv'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return program


def cubemap_to_dual_paraboloid(cube_faces):
    _, height, width, n_channels = cube_faces.shape
    height, width = height * 2, width * 2
    internal_format = 'rgba32f' if n_channels == 4 else 'rgb32f'

    with ContextProvider((height, width)):
        rendtex = gloo.Texture2D(
            (height, width, n_channels), interpolation='linear',
            wrapping='repeat', internalformat=internal_format)

        framebuffer = gloo.FrameBuffer(
            rendtex, gloo.RenderBuffer((width, height, n_channels)))

        program = CubemapToDualParaboloidProgram().compile()
        program['u_cubemap'] = gloo.TextureCubeMap(
            cube_faces, internalformat=internal_format, mipmap_levels=8)

        results = []
        for i in [0, 1]:
            with framebuffer:
                gloo.set_viewport(0, 0, width, height)
                program['u_hemisphere'] = i
                program.draw(gl.GL_TRIANGLE_STRIP)
                results.append(gloo.read_pixels(out_type=np.float32,
                                                format='rgb'))

    return results[0], results[1]


def _set_grid(grid: np.ndarray, height, width, u, v, value):
    grid[u*height:(u+1)*height, v*width:(v+1)*width] = value


def _get_grid(grid, height, width, u, v):
    return grid[u*height:(u+1)*height, v*width:(v+1)*width]


def stack_cross(cube_faces: np.ndarray, format='vertical'):
    _, height, width = cube_faces.shape[:3]
    n_channels = cube_faces.shape[3] if len(cube_faces.shape) == 4 else 1
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

    n_channels = cross.shape[2] if len(cross.shape) == 3 else 1
    faces_shape = ((6, height, width, n_channels)
                   if n_channels > 1 else (6, height, width))
    faces = np.zeros(faces_shape, dtype=np.float32)
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


def load_cubemap(path, size=(512, 512)):
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


class RadianceMap():
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
