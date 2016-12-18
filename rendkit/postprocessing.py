from vispy import gloo

from rendkit.glsl import GLSLTemplate, GLSLProgram


class IdentityProgram(GLSLProgram):
    def __init__(self, rendtex: gloo.Texture2D):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('postprocessing/identity.frag.glsl'))
        self.rendtex = rendtex

    def update_uniforms(self, program):
        program['a_texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['u_rendtex'] = self.rendtex
        return program


class SSAAProgram(GLSLProgram):
    MAX_SCALE = 3

    def __init__(self, rendtex: gloo.Texture2D, scale: int):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('postprocessing/ssaa.frag.glsl'))
        self.rendtex = rendtex
        self.scale = scale

    def update_uniforms(self, program):
        program['u_rendtex'] = self.rendtex
        program['u_texture_shape'] = self.rendtex.shape[:2]
        program['u_aa_kernel'] = [
            None,
            None,
            [0.44031130485056913, 0.29880437751590694,
             0.04535643028360444, -0.06431646022479595],
            [0.2797564513818748, 0.2310717037833796,
             0.11797652759318597, 0.01107354293249700],
        ][self.scale]
        program['a_texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        return program



class GammaCorrectionProgram(GLSLProgram):
    def __init__(self, rendtex, gamma=2.2):
        super().__init__(
            GLSLTemplate.fromfile('postprocessing/quad.vert.glsl'),
            GLSLTemplate.fromfile('postprocessing/gamma_correction.frag.glsl'))
        self.rendtex = rendtex
        self.gamma = gamma

    def update_uniforms(self, program):
        program['a_texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        program['u_rendtex'] = self.rendtex
        program['u_gamma'] = self.gamma
        return program
