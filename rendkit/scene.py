from collections import OrderedDict

import numpy as np
from scipy import misc
from typing import List, Dict

import rendkit.envmap
from meshkit import Mesh
from rendkit import vector_utils, util
from rendkit.camera import OrthographicCamera
from rendkit.core import logger, DepthRenderer, mesh_to_renderables
from rendkit.glsl import GLSLProgram
from rendkit.lights import Light
from rendkit.materials import PLACEHOLDER_MATERIAL
from vispy import gloo
from vispy.util import transforms


class Scene:
    def __init__(self,
                 lights: List[Light]=None,
                 materials: Dict[str, GLSLProgram]=None):
        self.lights = [] if lights is None else lights
        self.radiance_map = None
        self.materials = {} if materials is None else materials
        self.meshes = []
        self.renderables_by_mesh = OrderedDict()
        self.shadow_sources = []
        self._version = 0

    def reset(self):
        for material in self.materials.values():
            material._instances = []
            material.init_uniforms()
        for renderable in self.renderables:
            renderable._program = None
        if self.radiance_map:
            self.radiance_map.reset()

    def get_material(self, name):
        if name in self.materials:
            return self.materials[name]
        logger.warning('Material {} is not defined! Rendering with'
                       ' placeholder'.format(name))
        return PLACEHOLDER_MATERIAL

    def put_material(self, name: str, material: GLSLProgram):
        if name in self.materials:
            logger.warning("Material {} is already defined, overwriting."
                           .format(name))
        self.materials[name] = material

    def add_light(self, light: Light):
        self.lights.append(light)
        self.mark_updated()

    def set_radiance_map(self, radiance_map, add_shadows=False):
        if radiance_map is None:
            return
        self.radiance_map = radiance_map

        if add_shadows:
            shadow_dirs = rendkit.envmap.prefilter.find_shadow_sources(
                self.radiance_map.radiance_faces)
            logger.info("Rendering {} shadow maps.".format(len(shadow_dirs)))

            with DepthRenderer() as r:
                for i, shadow_dir in enumerate(shadow_dirs):
                    position = vector_utils.normalized(shadow_dir)
                    up = np.roll(position, 1) * (1, 1, -1)
                    camera = OrthographicCamera(
                        (200, 200), -150, 150, position=position,
                        lookat=(0, 0, 0), up=up)
                    rend_target = util.create_rend_target((1024, 1024))
                    r.draw(camera, self.renderables, rend_target)
                    with rend_target[0]:
                        # gloo.read_pixels flips the output.
                        depth = np.flipud(np.copy(gloo.read_pixels(
                            format='depth', out_type=np.float32)))
                        # Opengl ES doesn't support border clamp value
                        # specification so hack it like this.
                        depth[[0, -1], :] = 1.0
                        depth[:, [0, -1]] = 1.0
                    self.shadow_sources.append((camera, depth))
                    misc.imsave('/srv/www/out/__{}.png'.format(i), depth)

        self.mark_updated()

    def add_mesh(self, mesh: Mesh, position=(0, 0, 0)):
        model_mat = transforms.translate(position).T
        self.meshes.append(mesh)
        self.renderables_by_mesh[mesh] = mesh_to_renderables(mesh, model_mat)

    def set_mesh_transform(self, mesh: Mesh, transform_mat: np.ndarray,
                           apply_to_existing=False):
        if transform_mat.shape != (4, 4):
            raise ValueError("Invalid transformation matrix (must be 4x4).")
        for renderable in self.renderables_by_mesh[mesh]:
            if apply_to_existing:
                renderable.model_mat = transform_mat @ renderable.model_mat
            else:
                renderable.model_mat = transform_mat

    @property
    def renderables(self):
        result = []
        for mesh_renderables in self.renderables_by_mesh.values():
            for renderable in mesh_renderables:
                result.append(renderable)
        return result

    @property
    def version(self):
        return self._version

    def mark_updated(self):
        self._version += 1
