import logging
from collections import OrderedDict
from typing import List, Dict

import numpy as np
from numpy import linalg
from vispy.util import transforms

from meshkit import Mesh
from rendkit.lights import Light, PointLight, DirectionalLight
from rendkit.materials import PLACEHOLDER_MATERIAL, GLSLProgram

logger = logging.getLogger(__name__)


class Renderable:
    def __init__(self,
                 material_name: str,
                 attributes: Dict[str, np.ndarray],
                 model_mat=np.eye(4)):
        self.model_mat = model_mat
        self.material_name = material_name
        self._attributes = attributes
        self._uv_scale = 1.0

        self._current_scene = None
        self._program = None

    def compile(self, scene):
        material = scene.get_material(self.material_name)
        program = material.compile(
            len(scene.lights), scene.radiance_map is not None)

        used_attributes = {'a_position'}
        if material.use_normals:
            used_attributes.add('a_normal')
        if material.use_tangents:
            used_attributes.add('a_tangent')
            used_attributes.add('a_bitangent')
        if material.use_uvs:
            used_attributes.add('a_uv')

        for a_name in used_attributes:
            program[a_name] = self._attributes[a_name].astype(np.float32)

        return program

    def scale_uvs(self, scale):
        if 'a_uv' in self._attributes:
            self._attributes['a_uv'] *= scale / self._uv_scale
            if self._program is not None:
                self._program['a_uv'] = np.float32(self._attributes['a_uv'])
        self._uv_scale = scale

    def activate(self, scene, camera):
        material = scene.get_material(self.material_name)
        if self._program is None or scene != self._current_scene:
            self._current_scene = scene
            self._program = self.compile(scene)
        program = self._program
        if material.use_lights:
            for i, light in enumerate(scene.lights):
                program['u_light_type[{}]'.format(i)] = light.type
                if (light.type == PointLight.type
                    or light.type == DirectionalLight.type):
                    program['u_light_position[{}]'.format(i)] = light.position
                    program['u_light_intensity[{}]'.format(i)] = light.intensity
                    program['u_light_color[{}]'.format(i)] = light.color

        if scene.radiance_map is not None and material.use_radiance_map:
            program['u_irradiance_map'] = scene.radiance_map.irradiance_tex
            program['u_cubemap_size'] = scene.radiance_map.size
            program['u_radiance_upper'] = scene.radiance_map.radiance_upper_tex
            program['u_radiance_lower'] = scene.radiance_map.radiance_lower_tex

        if material.use_cam_pos:
            program['u_cam_pos'] = linalg.inv(camera.view_mat())[:3, 3]
        program['u_view'] = camera.view_mat().T
        program['u_model'] = self.model_mat.T
        program['u_projection'] = camera.projection_mat().T
        if material.use_near_far:
            program['u_near'] = -camera.near
            program['u_far'] = -camera.far
        return program


class Scene:
    def __init__(self,
                 lights: List[Light]=None,
                 materials: Dict[str, GLSLProgram]=None):
        self.lights = [] if lights is None else lights
        self.radiance_map = None
        self.materials = {} if materials is None else materials
        self.meshes = []
        self.renderables_by_mesh = OrderedDict()
        self._version = 0

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
        self._version += 1

    def set_radiance_map(self, radiance_map):
        self.radiance_map = radiance_map
        self._version += 1

    def add_mesh(self, mesh: Mesh, position=(0, 0, 0)):
        model_mat = transforms.translate(position).T
        self.meshes.append(mesh)
        self.renderables_by_mesh[mesh] = mesh_to_renderables(mesh, model_mat)

    @property
    def renderables(self):
        for mesh_renderables in self.renderables_by_mesh.values():
            for renderable in mesh_renderables:
                yield renderable


def mesh_to_renderables(mesh: Mesh, model_mat):
    renderables = []
    # For now each renderable represents a submesh with the same materials.
    for material_id, material_name in enumerate(mesh.materials):
        filter = {'material': material_id}
        vertex_positions = mesh.expand_face_vertices(filter)
        vertex_normals = mesh.expand_face_normals(filter)
        vertex_tangents, vertex_bitangents = mesh.expand_tangents(
            filter)
        vertex_uvs = mesh.expand_face_uvs(filter)
        if len(vertex_positions) < 3:
            logger.warning('Material {} not visible.'.format(material_name))
            continue
        attributes = dict(
            a_position=vertex_positions,
            a_normal=vertex_normals,
            a_tangent=vertex_tangents,
            a_bitangent=vertex_bitangents,
            a_uv=vertex_uvs
        )
        renderables.append(Renderable(material_name, attributes, model_mat))
    return renderables
