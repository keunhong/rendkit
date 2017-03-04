import os
import copy
import logging
from typing import Dict, List, Union

import numpy as np
from vispy import gloo
from vispy.gloo import gl

import rendkit.materials
from meshkit import Mesh
from meshkit import wavefront
from rendkit import cubemap as cm
from rendkit.lights import Light, PointLight, DirectionalLight
from rendkit.cubemap import RadianceMap
from rendkit.materials import (GLSLProgram, SVBRDFMaterial, PhongMaterial,
                               BasicMaterial, NormalMaterial, WorldCoordMaterial,
                               DepthMaterial, UVMaterial, UnwrapToUVMaterial,
                               TangentMaterial, BitangentMaterial)
from rendkit import postprocessing as pp
from svbrdf import SVBRDF
from .camera import CalibratedCamera, PerspectiveCamera, ArcballCamera
from .core import Scene
from rendkit.renderers import Renderer


class _nop():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


logger = logging.getLogger(__name__)


class JSDRenderer(Renderer):

    def __init__(self, jsd_dict_or_scene, camera=None, size=None,
                 conservative_raster=False,
                 gamma=None,
                 ssaa=0,
                 exposure=1.0,
                 *args, **kwargs):
        if camera is None:
            camera = import_jsd_camera(jsd_dict_or_scene)
        scene = jsd_dict_or_scene
        if isinstance(scene, dict):
            scene = import_jsd_scene(jsd_dict_or_scene)

        super().__init__(size, camera, scene, *args, **kwargs)
        gloo.set_state(depth_test=True)
        if conservative_raster:
            from . import nvidia
            self.conservative_raster = nvidia.conservative_raster(True)
        else:
            self.conservative_raster = _nop()
        self.gamma = gamma
        self.ssaa = min(max(1, ssaa), pp.SSAAProgram.MAX_SCALE)

        # Create original rendering buffer.
        self.render_size = (self.size[0] * self.ssaa, self.size[1] * self.ssaa)
        self.pp_pipeline = pp.PostprocessPipeline(self.size, self)
        self.render_tex, self.render_fb = pp.create_rend_target(self.render_size)

        logger.info("Render size: {} --SSAAx{}--> {}".format(
            self.size, self.ssaa, self.render_size))

        if self.ssaa > 1:
            self.pp_pipeline.add_program(pp.SSAAProgram(ssaa))
        if gamma is not None:
            logger.info("Post-processing gamma={}, exposure={}."
                        .format(gamma, exposure))
            self.pp_pipeline.add_program(pp.ExposureProgram(exposure))
            self.pp_pipeline.add_program(pp.GammaCorrectionProgram(gamma))
        else:
            self.pp_pipeline.add_program(pp.IdentityProgram())

    def draw(self):
        with self.render_fb:
            gloo.clear(color=self.camera.clear_color)
            gloo.set_state(depth_test=True)
            gloo.set_viewport(0, 0, *self.render_size)
            for renderable in self.scene.renderables:
                program = renderable.activate(self.scene, self.camera)
                with self.conservative_raster:
                    program.draw(gl.GL_TRIANGLES)

        self.pp_pipeline.draw(self.render_tex)


def import_jsd_scene(jsd_dict):
    scene = Scene(lights=import_jsd_lights(jsd_dict),
                  materials=import_jsd_materials(jsd_dict))
    scene.add_mesh(import_jsd_mesh(jsd_dict["mesh"]))
    scene.set_radiance_map(import_radiance_map(jsd_dict))
    return scene


def import_jsd_camera(jsd_dict):
    if 'camera' in jsd_dict:
        jsd_cam = jsd_dict['camera']
        type = jsd_cam['type']
        clear_color = jsd_cam.get('clear_color', (1.0, 1.0, 1.0))
        if type == 'perspective':
            return PerspectiveCamera(
                size=jsd_cam['size'],
                near=jsd_cam['near'],
                far=jsd_cam['far'],
                fov=jsd_cam['fov'],
                position=jsd_cam['position'],
                lookat=jsd_cam['lookat'],
                up=jsd_cam['up'],
                clear_color=clear_color)
        elif type == 'arcball':
            return ArcballCamera(
                size=jsd_cam['size'],
                near=jsd_cam['near'],
                far=jsd_cam['far'],
                fov=jsd_cam['fov'],
                position=jsd_cam['position'],
                lookat=jsd_cam['lookat'],
                up=jsd_cam['up'],
                clear_color=clear_color)
        elif type == 'calibrated':
            return CalibratedCamera(
                size=jsd_cam['size'],
                near=jsd_cam['near'],
                far=jsd_cam['far'],
                extrinsic=np.array(jsd_cam['extrinsic']),
                intrinsic=np.array(jsd_cam['intrinsic']),
                clear_color=clear_color)
        else:
            raise RuntimeError('Unknown camera type {}'.format(type))

    logger.warning('Camera undefined, returning default camera.')
    return ArcballCamera(
        size=(1024, 768), fov=75, near=10, far=1000.0,
        position=[50, 50, 50],
        lookat=(0.0, 0.0, -0.0),
        up=(0.0, 1.0, 0.0))


def import_jsd_lights(jsd_dict) -> List[Light]:
    lights = []
    jsd_lights = jsd_dict.get('lights', None)
    if jsd_lights is None:
        logger.warning("No lights defined, using default lights")
        jsd_lights = [
            {
                "type": "directional",
                "position": [50, 100, -50],
                "intensity": 1.0
            },
        ]

    for jsd_light in jsd_lights:
        lights.append(import_jsd_light(jsd_light))

    return lights


def import_radiance_map(jsd_dict) -> Union[RadianceMap, None]:
    if 'radiance_map' not in jsd_dict:
        return None
    jsd_radmap = jsd_dict['radiance_map']
    scale = jsd_radmap['scale'] if ('scale' in jsd_radmap) else 1.0

    if 'path' in jsd_radmap:
        path = jsd_radmap['path']
        logger.info('Importing radiance map from {} with scale={}'
                    .format(path, scale))
        cube_faces = cm.load_cubemap(path)
    elif jsd_radmap['type'] == 'inline':
        logger.info("Importing inline radiance map with shape={}".format(
            jsd_radmap['array'].shape))
        cube_faces = np.array(jsd_radmap['array'], dtype=np.float32)
    else:
        raise RuntimeError('Unknown radiance map type {}!'.format(
            jsd_radmap['type']))
    assert cube_faces.shape[0] == 6
    if 'max' in jsd_radmap:
        cube_faces = np.clip(cube_faces, 0, jsd_radmap['max'])
    logger.info('Radiance range: ({}, {})'
                .format(cube_faces.min(), cube_faces.max()))
    return RadianceMap(cube_faces, scale)


def import_jsd_light(jsd_light) -> Light:
    if jsd_light['type'] == 'point':
        return PointLight(jsd_light['position'],
                          jsd_light['intensity'])
    elif jsd_light['type'] == 'directional':
        return DirectionalLight(jsd_light['position'],
                                jsd_light['intensity'])
    else:
        raise RuntimeError('Unknown light type {}'.format(jsd_light['type']))


def list_material_names(jsd_dict):
    return [m for m in jsd_dict['materials'].keys()]


def import_jsd_materials(jsd_dict) -> Dict[str, GLSLProgram]:
    materials = {}
    for name, jsd_material in jsd_dict["materials"].items():
        materials[name] = import_jsd_material(jsd_material)
    return materials


def import_jsd_material(jsd_material) -> rendkit.materials.GLSLProgram:
    if jsd_material['type'] == 'svbrdf':
        transposed = False
        if 'transposed' in jsd_material:
            transposed = bool(jsd_material['transposed'])
        return SVBRDFMaterial(SVBRDF(jsd_material['path'], transposed=transposed))
    elif jsd_material['type'] == 'basic':
        return BasicMaterial(jsd_material['color'])
    elif jsd_material['type'] == 'svbrdf_inline':
        return SVBRDFMaterial(SVBRDF(
            diffuse_map=jsd_material['diffuse_map'],
            specular_map=jsd_material['specular_map'],
            spec_shape_map=jsd_material['spec_shape_map'],
            normal_map=jsd_material['normal_map'],
            alpha=jsd_material['alpha']))
    elif jsd_material['type'] == 'phong':
        return PhongMaterial(
            jsd_material['diffuse'],
            jsd_material['specular'],
            jsd_material['shininess'])
    elif jsd_material['type'] == 'uv':
        return UVMaterial()
    elif jsd_material['type'] == 'depth':
        return DepthMaterial()
    elif jsd_material['type'] == 'normal':
        return NormalMaterial()
    elif jsd_material['type'] == 'tangent':
        return TangentMaterial()
    elif jsd_material['type'] == 'bitangent':
        return BitangentMaterial()
    elif jsd_material['type'] == 'world':
        return WorldCoordMaterial()
    elif jsd_material['type'] == 'unwrap_to_uv':
        return UnwrapToUVMaterial(
            jsd_material['image'],
            jsd_material['depth'])


def import_jsd_mesh(jsd_mesh):
    if jsd_mesh['type'] == 'wavefront':
        mesh = wavefront.read_obj_file(jsd_mesh['path'])
    elif jsd_mesh['type'] == 'inline':
        vertices = jsd_mesh['vertices']
        normals = jsd_mesh['normals'] if 'normals' in jsd_mesh else []
        uvs = jsd_mesh['uvs'] if 'uvs' in jsd_mesh else []
        mesh = Mesh(np.array(vertices),
                    np.array(normals),
                    np.array(uvs),
                    jsd_mesh['faces'],
                    jsd_mesh['materials'], [], [])
    else:
        raise RuntimeError("Unknown mesh type {}".format(jsd_mesh['type']))

    if 'size' in jsd_mesh:
        mesh.resize(jsd_mesh['size'])
    elif 'scale' in jsd_mesh:
        mesh.rescale(jsd_mesh['scale'])

    if 'uv_scale' in jsd_mesh:
        uv_scale = float(jsd_mesh['uv_scale'])
        logger.info("UV scale is set to {:.04f} for mesh"
                    .format(uv_scale))
        mesh.uvs *= float(uv_scale)

    return mesh


def export_mesh_to_jsd(mesh: Mesh, size=100):
    return {
        "size": size,
        "type": "inline",
        "vertices": mesh.vertices.tolist(),
        "uvs": mesh.uvs.tolist() if len(mesh.uvs) > 0 else [],
        "normals": mesh.normals.tolist(),
        "materials": list(mesh.materials),
        "faces": copy.copy(mesh.faces),
    }
