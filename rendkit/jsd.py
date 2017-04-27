import copy
import logging
from typing import Dict, List, Union

import numpy as np

import rendkit.materials
from meshkit import Mesh
from meshkit import wavefront
from rendkit import shapes
from rendkit.envmap import EnvironmentMap
from rendkit.envmap.io import load_envmap
from rendkit.lights import Light, PointLight, DirectionalLight
from rendkit.materials import (GLSLProgram, SVBRDFMaterial, PhongMaterial,
                               BasicMaterial, NormalMaterial,
                               WorldCoordMaterial,
                               DepthMaterial, UVMaterial, UnwrapToUVMaterial,
                               TangentMaterial, BitangentMaterial,
                               BasicTextureMaterial)
from rendkit.renderers import SceneRenderer
from rendkit.scene import Scene
from svbrdf import SVBRDF
from .camera import CalibratedCamera, PerspectiveCamera, ArcballCamera

logger = logging.getLogger(__name__)


class JSDRenderer(SceneRenderer):
    def __init__(self, jsd_dict_or_scene, camera=None, show_floor=False,
                 shadows=False, *args, **kwargs):
        if camera is None:
            camera = import_jsd_camera(jsd_dict_or_scene)
        scene = jsd_dict_or_scene
        if isinstance(scene, dict):
            jsd_dict = jsd_dict_or_scene
            scene = Scene(lights=import_jsd_lights(jsd_dict),
                          materials=import_jsd_materials(jsd_dict))
            scene.add_mesh(import_jsd_mesh(jsd_dict["mesh"]))

            if show_floor:
                floor_pos = scene.meshes[0].vertices[:, 1].min()
                floor_mesh = shapes.make_plane(10000, 10000, 'floor')
                scene.add_mesh(floor_mesh, (0, floor_pos, 0))
                scene.put_material(
                    'floor', PhongMaterial((1.0, 1.0, 1.0),
                                           (0.00, 0.00, 0.00), 500.0))

            if 'radiance_map' in jsd_dict:
                scene.set_radiance_map(import_radiance_map(
                    jsd_dict['radiance_map']),
                    add_shadows=shadows)

        super().__init__(scene=scene, camera=camera, *args, **kwargs)


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
    jsd_lights = jsd_dict.get('lights', None)
    if jsd_lights is None:
        return []

    lights = []
    for jsd_light in jsd_lights:
        lights.append(import_jsd_light(jsd_light))

    return lights


def import_radiance_map(jsd_radmap) -> Union[EnvironmentMap, None]:
    scale = jsd_radmap['scale'] if ('scale' in jsd_radmap) else 1.0

    if 'path' in jsd_radmap:
        path = jsd_radmap['path']
        logger.info('Importing radiance map from {} with scale={}'
                    .format(path, scale))
        cube_faces = load_envmap(path)
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
    return EnvironmentMap(cube_faces, scale)


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
            alpha=jsd_material['alpha'],
            cdf_sampler=jsd_material.get('cdf_sampler', None),
            pdf_sampler=jsd_material.get('pdf_sampler', None),
            sigma_min=jsd_material.get('sigma_min', None),
            sigma_max=jsd_material.get('sigma_max', None)))
    elif jsd_material['type'] == 'basic_texture':
        return BasicTextureMaterial(jsd_material['texture'])
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
        mesh.uv_scale = uv_scale

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


def cache_inline(jsd_dict):
    new_dict = copy.deepcopy(jsd_dict)
    for mat_name, jsd_mat in new_dict['materials'].items():
        if jsd_mat['type'] == 'svbrdf':
            brdf = SVBRDF(jsd_mat['path'])
            new_dict['materials'][mat_name] = brdf.to_jsd()
    return new_dict
