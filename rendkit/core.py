import logging
from typing import List, Dict

import numpy as np
from numpy import linalg

from meshkit import Mesh
from rendkit.lights import Light, PointLight, DirectionalLight
from rendkit.materials import PLACEHOLDER_MATERIAL, GLSLProgram

logger = logging.getLogger(__name__)


class Renderable:
    def __init__(self,
                 material: GLSLProgram,
                 attributes: Dict[str, np.ndarray]):
        self.material = material
        self._attributes = attributes
        self._uv_scale = 1.0

        self._current_scene = None
        self._program = None

    def compile(self, scene):
        program = self.material.compile(len(scene.lights),
                                        scene.radiance_map is not None)
        for k, v in self._attributes.items():
            program[k] = v.astype(np.float32)

        return program

    def scale_uvs(self, scale):
        if 'a_uv' in self._attributes:
            self._attributes['a_uv'] *= scale / self._uv_scale
            if self._program is not None:
                self._program['a_uv'] = np.float32(self._attributes['a_uv'])
        self._uv_scale = scale

    def activate(self, scene, camera):
        if self._program is None or scene != self._current_scene:
            self._current_scene = scene
            self._program = self.compile(scene)
        program = self._program
        if self.material.use_lights:
            for i, light in enumerate(scene.lights):
                program['u_light_type[{}]'.format(i)] = light.type
                if (light.type == PointLight.type
                    or light.type == DirectionalLight.type):
                    program['u_light_position[{}]'.format(i)] = light.position
                    program['u_light_intensity[{}]'.format(i)] = light.intensity
                    program['u_light_color[{}]'.format(i)] = light.color

        if scene.radiance_map is not None and self.material.use_radiance_map:
            program['u_irradiance_map'] = scene.radiance_map.irradiance_tex
            program['u_radiance_map'] = scene.radiance_map.radiance_tex

        if self.material.use_cam_pos:
            program['u_cam_pos'] = linalg.inv(camera.view_mat())[:3, 3]
        program['u_view'] = camera.view_mat().T
        program['u_model'] = np.eye(4)
        program['u_projection'] = camera.projection_mat().T
        if self.material.use_near_far:
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
        self.renderables = []
        self._program_version = 0

    def put_material(self, name: str, material: GLSLProgram):
        if name in self.materials:
            logger.warn("Material {} is already defined, overwriting.".format(
                name))
        self.materials[name] = material

    def add_light(self, light: Light):
        self.lights.append(light)
        self._program_version += 1

    def set_radiance_map(self, radiance_map):
        self.radiance_map = radiance_map
        self._program_version += 1

    def add_mesh(self, mesh: Mesh):
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
            if material_name in self.materials:
                material = self.materials[material_name]
            else:
                logger.warning('Material {} is not defined! '
                      'Rendering with placeholder'.format(material_name))
                material = PLACEHOLDER_MATERIAL
            attributes = {
                'a_position': vertex_positions,
            }
            if material.use_normals:
                attributes['a_normal'] = vertex_normals
            if material.use_tangents:
                attributes['a_tangent'] = vertex_tangents
                attributes['a_bitangent'] = vertex_bitangents
            if material.use_uvs:
                attributes['a_uv'] = vertex_uvs
            self._add_renderable(Renderable(material, attributes))

    def _add_renderable(self, renderable: Renderable):
        self.renderables.append(renderable)
