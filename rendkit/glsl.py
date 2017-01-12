import os
from string import Template

from vispy import gloo


_package_dir = os.path.dirname(os.path.realpath(__file__))
_shader_dir = os.path.join(_package_dir, 'shaders')


class GLSLTemplate(Template):
    delimiter = 'TPL.'

    @classmethod
    def fromfile(cls, filename):
        path = os.path.join(_shader_dir, filename)
        with open(path, 'r') as f:
            return GLSLTemplate(f.read())


def _glsl_bool(val: bool) -> int:
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
        self._instance = None

    def compile(self, num_lights=0, use_radiance_map=False):
        use_radiance_map = use_radiance_map and self.use_radiance_map
        vs = self._vert_shader.substitute(
            use_normals=_glsl_bool(self.use_normals),
            use_tangents=_glsl_bool(self.use_tangents))
        fs = self._frag_shader.substitute(
            num_lights=num_lights,
            use_radiance_map=_glsl_bool(use_radiance_map))
        program = gloo.Program(vs, fs)
        program = self.update_uniforms(program)
        return program

    def update_uniforms(self, program):
        raise NotImplementedError
