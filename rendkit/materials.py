import numpy as np
from vispy.gloo import Texture2D

from rendkit.glsl import GLSLProgram, GLSLTemplate


class BasicMaterial(GLSLProgram):
    def __init__(self, color: np.ndarray):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('basic.frag.glsl'),
                         use_uvs=False,
                         use_lights=False,
                         use_cam_pos=False,
                         use_normals=False)
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


class SVBRDFMaterial(GLSLProgram):
    def __init__(self, svbrdf):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('svbrdf.frag.glsl'),
                         use_uvs=True,
                         use_cam_pos=True,
                         use_lights=True,
                         use_normals=True,
                         use_radiance_map=True)
        self.alpha = svbrdf.alpha
        self.diff_map = Texture2D(svbrdf.diffuse_map,
                                  interpolation='linear',
                                  wrapping='repeat',
                                  internalformat='rgb32f')
        self.spec_map = Texture2D(svbrdf.specular_map,
                                  interpolation='linear',
                                  wrapping='repeat',
                                  internalformat='rgb32f')
        self.spec_shape_map = Texture2D(svbrdf.spec_shape_map,
                                        wrapping='repeat',
                                        internalformat='rgb32f')
        self.normal_map = Texture2D(svbrdf.normal_map,
                                    interpolation='linear',
                                    wrapping='repeat',
                                    internalformat='rgb32f')

    def update_uniforms(self, program):
        program['alpha'] = self.alpha
        program['diff_map'] = self.diff_map
        program['spec_map'] = self.spec_map
        program['spec_shape_map'] = self.spec_shape_map
        program['normal_map'] = self.normal_map
        return program


class TweakableSVBRDFMaterial(GLSLProgram):

    def __init__(self, svbrdf):
        super().__init__(GLSLTemplate.fromfile('default.vert.glsl'),
                         GLSLTemplate.fromfile('svbrdf_tweakable.frag.glsl'),
                         use_uvs=True,
                         use_cam_pos=True,
                         use_lights=True,
                         use_normals=True)
        from skimage import color
        print('Converting diffuse map to Lab')
        diff_map_lab = np.clip(svbrdf.diffuse_map, 0, 1)
        diff_map_lab = color.rgb2lab(diff_map_lab).astype(dtype=np.float32)
        self.diff_map_mean = diff_map_lab.mean(axis=(0, 1))
        self.diff_map_std = diff_map_lab.std(axis=(0, 1))
        diff_map_lab = (diff_map_lab - self.diff_map_mean) / self.diff_map_std
        self.spec_scale = 1
        self.spec_shape_scale = 1
        self.uv_scale = 1.0

        self.alpha = svbrdf.alpha
        self.diff_map = Texture2D(diff_map_lab,
                                  interpolation='linear',
                                  wrapping='repeat',
                                  internalformat='rgb32f')
        self.spec_map = Texture2D(svbrdf.specular_map,
                                  interpolation='linear',
                                  wrapping='repeat',
                                  internalformat='rgb32f')
        self.spec_shape_map = Texture2D(svbrdf.spec_shape_map,
                                        wrapping='repeat',
                                        internalformat='rgb32f')
        self.normal_map = Texture2D(svbrdf.normal_map,
                                    interpolation='linear',
                                    wrapping='repeat',
                                    internalformat='rgb32f')

    def update_uniforms(self, program):
        program['alpha'] = self.alpha
        program['diff_map'] = self.diff_map
        program['spec_map'] = self.spec_map
        program['spec_shape_map'] = self.spec_shape_map
        program['normal_map'] = self.normal_map
        program['source_mean'] = np.clip(self.diff_map_mean, 0.0, 128.0)
        program['source_std'] = self.diff_map_std
        program['spec_scale'] = self.spec_scale
        program['spec_shape_scale'] = self.spec_shape_scale
        program['uv_scale'] = self.uv_scale
        return program


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
