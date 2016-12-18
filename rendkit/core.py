import logging
from typing import List, Dict, Tuple

import numpy as np
from numpy import linalg
from vispy import gloo, app

from meshkit import Mesh
from rendkit import vector_utils
from rendkit.camera import BaseCamera
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
                self._program['a_uv'] = self._attributes['a_uv']
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
            program['u_radiance_map'] = scene.radiance_map.texture
            program['u_radiance_map_size'] = scene.radiance_map.size[:2]

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
            if material.use_normals and material.use_uvs:
                attributes['a_tangent'] = vertex_tangents
                attributes['a_bitangent'] = vertex_bitangents
            if material.use_uvs:
                attributes['a_uv'] = vertex_uvs
            self._add_renderable(Renderable(material, attributes))

    def _add_renderable(self, renderable: Renderable):
        self.renderables.append(renderable)


class Renderer(app.Canvas):
    def __init__(self,
                 scene: Scene,
                 size: Tuple[int, int],
                 camera: BaseCamera, *args, **kwargs):
        if size is None:
            size = camera.size
        self.camera = camera
        super().__init__(size=size, *args, **kwargs)
        gloo.set_state(depth_test=True)
        gloo.set_viewport(0, 0, *self.size)

        self.scene = scene
        self.active_program = None
        self.size = size

        # Buffer shapes are HxW, not WxH...
        self._rendertex = gloo.Texture2D(
            shape=(size[1], size[0]) + (4,),
            internalformat='rgba32f')
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(
            shape=(size[1], size[0])))

        self.mesh = None

        self._current_material = None

    def set_program(self, vertex_shader, fragment_shader):
        self.active_program = gloo.Program(vertex_shader, fragment_shader)

    def draw(self):
        """
        Override and implement drawing logic here. e.g. gloo.clear_color
        """
        raise NotImplementedError

    def render_to_image(self) -> np.ndarray:
        """
        Renders to an image.
        :return: image of rendered scene.
        """
        with self._fbo:
            self.draw()
            pixels = gloo.util.read_pixels(out_type=np.float32,
                                           alpha=False)
        return pixels

    def on_resize(self, event):
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.camera.size = self.size

    def on_draw(self, event):
        self.draw()

    def on_mouse_move(self, event):
        if event.is_dragging:
            self.camera.handle_mouse(event.last_event.pos, event.pos)
            self.update()

    def on_mouse_wheel(self, event):
        cur_dist = linalg.norm(self.camera.position)
        zoom_amount = 5.0 * event.delta[1]
        zoom_dir = vector_utils.normalized(self.camera.position)
        if cur_dist - zoom_amount > 0:
            self.camera.position -= zoom_amount * zoom_dir
            self.update()

    def __enter__(self):
        self._backend._vispy_warmup()
        return self
