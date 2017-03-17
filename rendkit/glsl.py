import os
from string import Template

import numpy as np
from numpy import linalg

from rendkit.lights import PointLight, DirectionalLight
from vispy.util import config as vispy_config
from vispy import gloo

_package_dir = os.path.dirname(os.path.realpath(__file__))
_shader_dir = os.path.join(_package_dir, 'shaders')


vispy_config['include_path'].append(_shader_dir)


class GLSLTemplate(Template):
    delimiter = 'TPL.'

    @classmethod
    def fromfile(cls, filename):
        path = os.path.join(_shader_dir, filename)
        with open(path, 'r') as f:
            return GLSLTemplate(f.read())


def glsl_bool(val: bool) -> int:
    return 1 if val else 0


class GLSLProgram:
    def __init__(self,
                 vert_shader: GLSLTemplate,
                 frag_shader: GLSLTemplate,
                 use_normals=False,
                 use_uvs=False,
                 use_cam_pos=False,
                 use_lights=True,
                 use_near_far=False,
                 use_radiance_map=False,
                 use_tangents=False):
        self.use_uvs = use_uvs
        self.use_cam_pos = use_cam_pos
        self.use_lights = use_lights
        self.use_normals = use_normals
        self.use_near_far = use_near_far
        self.use_radiance_map = use_radiance_map
        self.use_tangents = use_tangents

        self._vert_shader = vert_shader
        self._frag_shader = frag_shader
        self.uniforms = {}

        self.vert_tpl_vars = {}
        self.frag_tpl_vars = {}
        self._instances = []

    def compile(self, num_lights=0, num_shadow_sources=0,
                use_radiance_map=False):
        use_radiance_map = use_radiance_map and self.use_radiance_map
        vs = self._vert_shader.substitute(
            use_normals=glsl_bool(self.use_normals),
            use_tangents=glsl_bool(self.use_tangents),
            num_shadow_sources=num_shadow_sources,
            use_radiance_map=glsl_bool(use_radiance_map),
            **self.vert_tpl_vars)
        fs = self._frag_shader.substitute(
            num_lights=num_lights,
            num_shadow_sources=num_shadow_sources,
            use_radiance_map=glsl_bool(use_radiance_map),
            **self.frag_tpl_vars)
        program = gloo.Program(vs, fs)
        self.upload_uniforms(program)
        self._instances.append(program)
        return program

    def upload_uniforms(self, program):
        """
        Uploads uniforms to the program instance.
        """
        for k, v in self.uniforms.items():
            program[k] = v

    def set_attributes(self, program, attributes):
        used_attributes = {'a_position'}
        if self.use_normals:
            used_attributes.add('a_normal')
        if self.use_tangents:
            used_attributes.add('a_tangent')
            used_attributes.add('a_bitangent')
        if self.use_uvs:
            used_attributes.add('a_uv')

        for a_name in used_attributes:
            program[a_name] = attributes[a_name].astype(np.float32)

    def set_lights(self, program, lights):
        if self.use_lights:
            for i, light in enumerate(lights):
                program['u_light_type[{}]'.format(i)] = light.type
                if (light.type == PointLight.type
                    or light.type == DirectionalLight.type):
                    program['u_light_position[{}]'.format(i)] = light.position
                    program['u_light_intensity[{}]'.format(i)] = light.intensity
                    program['u_light_color[{}]'.format(i)] = light.color

    def set_radmap(self, program, radmap):
        if radmap is not None and self.use_radiance_map:
            program['u_irradiance_map'] = radmap.irradiance_tex
            program['u_cubemap_size'] = radmap.size
            program['u_radiance_upper'] = radmap.radiance_upper_tex
            program['u_radiance_lower'] = radmap.radiance_lower_tex

    def set_shadow_sources(self, program, shadow_sources):
        if self.use_radiance_map:
            for i, (shadow_cam, shadow_depth) in enumerate(shadow_sources):
                program['u_shadow_depth[{}]'.format(i)] = \
                    gloo.Texture2D(shadow_depth,
                                   interpolation='linear',
                                   wrapping='clamp_to_edge')
                program['u_shadow_proj[{}]'.format(i)] = shadow_cam.projection_mat().T
                program['u_shadow_view[{}]'.format(i)] = shadow_cam.view_mat().T

    def set_camera(self, program, camera):
        if self.use_cam_pos:
            program['u_cam_pos'] = linalg.inv(camera.view_mat())[:3, 3]
        program['u_view'] = camera.view_mat().T
        program['u_projection'] = camera.projection_mat().T
        if self.use_near_far:
            program['u_near'] = -camera.near
            program['u_far'] = -camera.far
