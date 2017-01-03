import copy
import logging
import math
from typing import Dict, List

import numpy as np
from scipy.misc import imread
from vispy import gloo
from vispy.gloo import gl

import rendkit.materials
from meshkit import wavefront, Mesh
from rendkit import pfm
from rendkit.lights import Light, PointLight, DirectionalLight, \
    RadianceMap
from rendkit.materials import (GLSLProgram, SVBRDFMaterial, PhongMaterial,
                               BasicMaterial, NormalMaterial, WorldCoordMaterial,
                               DepthMaterial, UVMaterial, UnwrapToUVMaterial)
from rendkit.postprocessing import GammaCorrectionProgram, IdentityProgram, \
    SSAAProgram
from svbrdf import SVBRDF
from .camera import CalibratedCamera, PerspectiveCamera, ArcballCamera
from .core import Renderer, Scene


class _nop():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


logger = logging.getLogger(__name__)


class JSDRenderer(Renderer):

    def __init__(self, jsd_dict, camera=None, size=None,
                 conservative_raster=False,
                 use_gamma_correction=False,
                 ssaa=0,
                 *args, **kwargs):
        if camera is None:
            camera = import_jsd_camera(jsd_dict)
        scene = import_jsd_scene(jsd_dict)
        super().__init__(scene, size, camera, *args, **kwargs)
        gloo.set_state(depth_test=True)
        if conservative_raster:
            from . import nvidia
            self.conservative_raster = nvidia.conservative_raster(True)
        else:
            self.conservative_raster = _nop()
        self.use_gamma_correction = use_gamma_correction
        self.ssaa = min(max(1, ssaa), SSAAProgram.MAX_SCALE)

        render_size = (self.size[0] * self.ssaa,
                       self.size[1] * self.ssaa)
        self.rendtex_size = ((2 ** (int(math.log2(render_size[0])) + 1)),
                        (2 ** (int(math.log2(render_size[1]) + 1))))
        logger.info("Render size: {}"
                    " --SSAAx{}--> {}"
                    " --pow of 2--> {}".format(
            self.size, 2**ssaa, render_size, self.rendtex_size))

        # Buffer shapes are HxW, not WxH. Shapes are HxW, sizes are WxH.
        rendtex_shape = (self.rendtex_size[1], self.rendtex_size[0])
        rendtex = gloo.Texture2D((*rendtex_shape, 4), interpolation='linear',
                                 internalformat='rgba32f')
        self.rendtex = rendtex
        self.rendfb = gloo.FrameBuffer(rendtex, gloo.RenderBuffer(rendtex_shape))
        self.pp_pipeline = []
        if self.ssaa > 1:
            logger.info("Post-processing SSAAx{} enabled: rendtex {} -> {}".format(
                ssaa, self.size, self.rendtex_size))
            self.pp_pipeline.append(SSAAProgram(rendtex, ssaa).compile())
        if use_gamma_correction:
            logger.info("Post-processing Gamma Correction enabled.")
            self.pp_pipeline.append(GammaCorrectionProgram(rendtex, gamma=2.2).compile())
        else:
            self.pp_pipeline.append(IdentityProgram(rendtex).compile())

    def draw(self):
        with self.rendfb:
            gloo.clear(color=self.camera.clear_color)
            gloo.set_state(depth_test=True)
            gloo.set_viewport(0, 0, *self.rendtex_size)
            for renderable in self.scene.renderables:
                program = renderable.activate(self.scene, self.camera)
                with self.conservative_raster:
                    program.draw(gl.GL_TRIANGLES)

        # Run postprocessing programs.
        for i, program in enumerate(self.pp_pipeline):
            is_last = i == len(self.pp_pipeline) - 1
            if is_last:
                gloo.set_viewport(0, 0, *self.physical_size)
                gloo.clear(color=True)
                gloo.set_state(depth_test=False)
            else:
                gloo.set_viewport(0, 0, *self.rendtex_size)

            with _nop() if is_last else self.rendfb:
                program['u_rendtex'] = self.rendtex
                program.draw(gl.GL_TRIANGLE_STRIP)


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
                "type": "point",
                "position": [150, -100, 0],
                "intensity": 3000
            },
            {
                "type": "point",
                "position": [70, 20, 70],
                "intensity": 3000
            },
            {
                "type": "point",
                "position": [0, 150, 0],
                "intensity": 3000
            }
        ]

    for jsd_light in jsd_lights:
        lights.append(import_jsd_light(jsd_light))

    return lights


def import_radiance_map(jsd_dict) -> RadianceMap:
    if 'radiance_map' not in jsd_dict:
        return None
    jsd_radmap = jsd_dict['radiance_map']
    scale = jsd_radmap['scale'] if ('scale' in jsd_radmap) else 1.0

    if 'type' not in jsd_radmap:
        raise RuntimeError('Radiance map type not specified')

    if jsd_radmap['type'] == 'pfm':
        logger.info('Importing radiance map from {} with scale={}'.format(
            jsd_radmap['path'], scale))
        array = pfm.load_pfm_texture(jsd_radmap['path'])
    elif jsd_radmap['type'] == 'image':
        logger.info('Importing radiance map from {} with scale={}'.format(
            jsd_radmap['path'], scale))
        array = imread(jsd_radmap['path'])
    elif jsd_radmap['type'] == 'inline':
        logger.info("Importing inline radiance map with shape={}".format(
            jsd_radmap['array'].shape))
        array = np.array(jsd_radmap['array'], dtype=np.float32)
    else:
        raise RuntimeError('Unknown radiance map type {}!'.format(
            jsd_radmap['type']))
    return RadianceMap(array, scale)


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
        return SVBRDFMaterial(SVBRDF(jsd_material['path']))
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
