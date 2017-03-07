import logging
from collections import OrderedDict
from typing import List, Dict

import numpy as np
from numpy import linalg
from scipy import misc

from rendkit import cubemap
from rendkit import util
from rendkit import vector_utils
from rendkit.camera import OrthographicCamera
from vispy import gloo, app
from vispy.gloo import gl
from vispy.util import transforms

from meshkit import Mesh
from rendkit.lights import Light, PointLight, DirectionalLight
from rendkit.materials import PLACEHOLDER_MATERIAL, GLSLProgram, DummyMaterial, \
    DepthMaterial

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
        self._scene_version = -1

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
            self._scene_version = -1
        if self._scene_version != scene.version:
            self._scene_version = scene.version
            self._program = material.compile(
                num_lights=len(scene.lights),
                num_shadow_sources=len(scene.shadow_sources),
                use_radiance_map=scene.radiance_map is not None)
            material.set_attributes(self._program, self._attributes)
            material.set_radmap(self._program, scene.radiance_map)
            material.set_shadow_sources(self._program, scene.shadow_sources)
            material.set_lights(self._program, scene.lights)

        material.set_camera(self._program, camera)
        self._program['u_model'] = self.model_mat.T

        return self._program


class DepthRenderer(app.Canvas):
    def __init__(self):
        super().__init__()
        self.material = DepthMaterial()
        self.program = DepthMaterial().compile()

    def draw(self, camera, renderables, rend_target):
        rendfb, rendtex, _ = rend_target

        with rendfb:
            gloo.clear(color=camera.clear_color)
            gloo.set_state(depth_test=True)
            gloo.set_viewport(0, 0, rendtex.shape[1], rendtex.shape[0])
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_FRONT)
            for renderable in renderables:
                self.material.set_camera(self.program, camera)
                self.material.set_attributes(self.program, renderable._attributes)
                self.program['u_model'] = renderable.model_mat.T
                self.program.draw(gl.GL_TRIANGLES)
            gl.glCullFace(gl.GL_BACK)


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
        if radiance_map is None:
            return
        self.radiance_map = radiance_map

        shadow_dirs = cubemap.find_shadow_sources(
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

    @property
    def version(self):
        return self._version


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
