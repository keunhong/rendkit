import numpy as np
from skimage import morphology as morph
from skimage.measure import regionprops

from rendkit import vector_utils
from rendkit.envmap.io import stack_cross, unstack_cross
from rendkit.glsl import GLSLProgram, GLSLTemplate
from rendkit.renderers import ContextProvider
from toolbox.images import rgb2gray
from vispy import gloo
from vispy.gloo import gl


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
    return [p[0] for p in shadow_positions[:num_shadows]]


class LambertPrefilterProgram(GLSLProgram):
    def __init__(self):
        super().__init__(
            GLSLTemplate.fromfile('cubemap/lambert.vert.glsl'),
            GLSLTemplate.fromfile('cubemap/lambert.frag.glsl'))

    def upload_uniforms(self, program):
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

    def upload_uniforms(self, program):
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
                results.append(np.flipud(gloo.read_pixels(out_type=np.float32,
                                                          format='rgb')))

    return results[0], results[1]
