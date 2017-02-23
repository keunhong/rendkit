import numpy as np
from scipy.special._ufuncs import gammaincinv

from vispy.gloo import Texture2D

from rendkit.glsl import GLSLProgram, GLSLTemplate


class BasicMaterial(GLSLProgram):
    def __init__(self, color: np.ndarray):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('basic.frag.glsl'),
                         use_uvs=False,
                         use_lights=False,
                         use_cam_pos=False,
                         use_normals=False,
                         use_tangents=False)
        self.color = color

    def update_uniforms(self, program):
        program['u_color'] = self.color
        return program


class PhongMaterial(GLSLProgram):
    def __init__(self, diff_color, spec_color, shininess):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('phong.frag.glsl'),
                         use_uvs=False,
                         use_lights=True,
                         use_cam_pos=True,
                         use_normals=True,
                         use_radiance_map=True)
        self.diff_color = diff_color
        self.spec_color = spec_color
        self.shininess = shininess

    def update_uniforms(self, program):
        program['u_diff'] = self.diff_color
        program['u_spec'] = self.spec_color
        program['u_shininess'] = self.shininess
        return program


class UVMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('uv.frag.glsl'),
                         use_uvs=True,
                         use_cam_pos=False,
                         use_lights=False,
                         use_normals=False)

    def update_uniforms(self, program):
        return program


class DepthMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('depth.vert.glsl'),
                         GLSLTemplate.fromfile('depth.frag.glsl'),
                         use_uvs=False,
                         use_cam_pos=False,
                         use_lights=False,
                         use_normals=False,
                         use_near_far=True)

    def update_uniforms(self, program):
        return program


class WorldCoordMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('world_coord.vert.glsl'),
                         GLSLTemplate.fromfile('world_coord.frag.glsl'),
                         use_uvs=False,
                         use_cam_pos=False,
                         use_lights=False,
                         use_normals=False)

    def update_uniforms(self, program):
        return program


class NormalMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('normal.frag.glsl'),
                         use_uvs=False,
                         use_cam_pos=False,
                         use_lights=False,
                         use_normals=True)

    def update_uniforms(self, program):
        return program


class TangentMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('tangent.frag.glsl'),
                         use_uvs=True,
                         use_cam_pos=False,
                         use_lights=False,
                         use_tangents=True)

    def update_uniforms(self, program):
        return program


class BitangentMaterial(GLSLProgram):
    def __init__(self):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('bitangent.frag.glsl'),
                         use_uvs=True,
                         use_cam_pos=False,
                         use_lights=False,
                         use_tangents=True)

    def update_uniforms(self, program):
        return program


class SVBRDFMaterial(GLSLProgram):
    def __init__(self, svbrdf):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('svbrdf.frag.glsl'),
                         use_uvs=True,
                         use_cam_pos=True,
                         use_lights=True,
                         use_normals=True,
                         use_tangents=True,
                         use_radiance_map=True)
        self.alpha = svbrdf.alpha
        self.diff_map = svbrdf.diffuse_map.astype(np.float32)
        self.spec_map = svbrdf.specular_map.astype(np.float32)
        self.spec_shape_map = svbrdf.spec_shape_map.astype(np.float32)
        self.normal_map = svbrdf.normal_map.astype(np.float32)

        S = self.spec_shape_map.reshape((-1, 3))
        S = S[:, [0, 2, 2, 1]].reshape((-1, 2, 2))
        trace = S[:, 0, 0] + S[:, 1, 1]
        beta = trace / 2
        self.sigma = beta ** (-1.0 / 4)
        x = np.linspace(0.0, 1, 500, endpoint=True)
        s = np.linspace(self.sigma.min(), self.sigma.max(), 500)
        self.theta_cdf = np.apply_along_axis(
            self.sample_cdf, 1, s[:, None], x=x).astype(dtype=np.float32)

    def update_uniforms(self, program):
        program['u_alpha'] = self.alpha
        program['u_diff_map'] = Texture2D(self.diff_map,
                                        interpolation='linear',
                                        wrapping='repeat',
                                        internalformat='rgb32f')
        program['u_spec_map'] = Texture2D(self.spec_map,
                                        interpolation='linear',
                                        wrapping='repeat',
                                        internalformat='rgb32f')
        program['u_spec_shape_map'] = Texture2D(self.spec_shape_map,
                                              interpolation='linear',
                                              wrapping='repeat',
                                              internalformat='rgb32f')
        program['u_normal_map'] = Texture2D(self.normal_map,
                                          interpolation='linear',
                                          wrapping='repeat',
                                          internalformat='rgb32f')
        program['u_theta_cdf'] = Texture2D(self.theta_cdf,
                                         interpolation='linear',
                                         wrapping='repeat',
                                         internalformat='r32f')
        program['u_sigma_range'] = (self.sigma.min(), self.sigma.max())
        return program

    def sample_cdf(self, sigma, x):
        alpha = self.alpha
        return np.arctan(sigma ** 2 * gammaincinv(1 / alpha, np.clip(
            1 - alpha / sigma ** 2 * x, 0, 1)) ** (1 / alpha))



class UnwrapToUVMaterial(GLSLProgram):
    def __init__(self, image, depth_im):
        super().__init__(GLSLTemplate.fromfile('unwrap_to_uv.vert.glsl'),
                         GLSLTemplate.fromfile('unwrap_to_uv.frag.glsl'),
                         use_uvs=True)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if depth_im.dtype != np.float32:
            depth_im = depth_im.astype(np.float32)
        self.input_tex = Texture2D(image,
                                   interpolation='linear',
                                   wrapping='clamp_to_edge',
                                   internalformat='rgb32f')
        self.input_depth = Texture2D(depth_im,
                                     interpolation='linear',
                                     wrapping='clamp_to_edge',
                                     internalformat='r32f')

    def update_uniforms(self, program):
        program['input_tex'] = self.input_tex
        program['input_depth'] = self.input_depth
        return program


PLACEHOLDER_MATERIAL = PhongMaterial([1.0, 0.0, 1.0], [0.1, 0.1, 0.1], 1.0)
