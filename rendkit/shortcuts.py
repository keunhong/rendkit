import math
import copy
from typing import List, Dict

import numpy as np

from meshkit import Mesh
from meshkit.wavefront import WavefrontMaterial
from rendkit import jsd
from toolbox import images
from toolbox.logging import init_logger
from .camera import ArcballCamera
from .jsd import JSDRenderer

logger = init_logger(__name__)


QUAL_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
    (166, 206, 227),
    (31, 120, 180),
    (178, 223, 138),
    (51, 160, 44),
    (251, 154, 153),
    (227, 26, 28),
    (253, 191, 111),
    (255, 127, 0),
    (202, 178, 214),
    (106, 61, 154),
    (255, 255, 153),
    (177, 89, 40),
    (141, 211, 199),
    (255, 255, 179),
    (190, 186, 218),
    (251, 128, 114),
    (128, 177, 211),
    (253, 180, 98),
    (179, 222, 105),
    (252, 205, 229),
    (217, 217, 217),
    (188, 128, 189),
    (204, 235, 197),
    (255, 237, 111),
]


def svbrdf_plane_renderer(svbrdf, size=None, lights=list(), radmap=None,
                          mode='all', gamma=2.2, uv_scale=1.0, shape=None,
                          transpose=False, camera=None,
                          cam_lookat=(0.0, 0.0), cam_fov=90,
                          cam_dist=1.0, cam_up=(1.0, 0.0, 0.0), **kwargs):
    if shape is None:
        height, width, _ = svbrdf.diffuse_map.shape
    else:
        height, width = shape[:2]
    zeros = np.zeros(svbrdf.diffuse_map.shape, dtype=np.float32)
    ones = np.ones(svbrdf.diffuse_map.shape, dtype=np.float32)
    if mode != 'all':
        svbrdf = copy.copy(svbrdf)
    if mode == 'diffuse_only':
        svbrdf.specular_map = zeros
    elif mode == 'specular_only':
        svbrdf.diffuse_map = zeros
    elif mode == 'light_map':
        gamma = None
        svbrdf.specular_map = zeros
        svbrdf.diffuse_map = ones

    if transpose:
        cam_up = (0.0, 0.0, -1.0)
    if transpose:
        cam_lookat = tuple(reversed(cam_lookat))

    if size is None:
        size = (width, height)
        cam_dist = cam_dist * min(width, height)/max(width, height)

    if camera is None:
        camera = ArcballCamera(
            size=size, fov=cam_fov, near=0.1, far=1000.0,
            position=[0, cam_dist, 0],
            lookat=(cam_lookat[0], 0.0, -cam_lookat[1]), up=cam_up)
    else:
        camera = camera

    plane_size = 100

    jsd = {
        "mesh": {
            "scale": 1,
            "type": "inline",
            "vertices": [
                [width/height*plane_size, 0.0, -plane_size],
                [width/height*plane_size, 0.0, plane_size],
                [-width/height*plane_size, 0.0, plane_size],
                [-width/height*plane_size, 0.0, -plane_size]
            ],
            "uvs": [
                [uv_scale*plane_size, 0.0],
                [uv_scale*plane_size, uv_scale*plane_size],
                [0.0, uv_scale*plane_size],
                [0.0, 0.0]
            ],
            "normals": [
                [0.0, 1.0, 0.0]
            ],
            "materials": ["plane"],
            "faces": [
                {
                    "vertices": [0, 1, 2],
                    "uvs": [0, 1, 2],
                    "normals": [0, 0, 0],
                    "material": 0
                },
                {
                    "vertices": [0, 2, 3],
                    "uvs": [0, 2, 3],
                    "normals": [0, 0, 0],
                    "material": 0
                }
            ]
        },
        "lights": lights,
        "materials": {
            "plane": {
                "type": "svbrdf_inline",
                "diffuse_map": svbrdf.diffuse_map,
                "specular_map": svbrdf.specular_map,
                "spec_shape_map": svbrdf.spec_shape_map,
                "normal_map": svbrdf.normal_map,
                "alpha": svbrdf.alpha,
            }
        }
    }
    if radmap is not None:
        if isinstance(radmap, np.ndarray):
            radmap = dict(type='inline', array=radmap)
        jsd['radiance_map'] = radmap
    return JSDRenderer(jsd, camera, size=size, gamma=gamma,
                       **kwargs)


def render_full(jsd_dict, **kwargs):
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        im = r.render_to_image()
    return im


def render_diffuse_lightmap(jsd_dict, **kwargs):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['diffuse_map'][:] = 1.0
            mat_jsd['specular_map'][:] = 0.0
        elif mat_jsd['type'] == 'beckmann_inline':
            mat_jsd['params']['diffuse_map'][:] = 1.0
            mat_jsd['params']['specular_map'][:] = 0.0
        else:
            raise RuntimeError('Unknown mat type.')
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_specular_lightmap(jsd_dict, **kwargs):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['diffuse_map'][:] = 0.0
            mat_jsd['specular_map'][:] = 1.0
        elif mat_jsd['type'] == 'beckmann_inline':
            mat_jsd['params']['diffuse_map'][:] = 0.0
            mat_jsd['params']['specular_map'][:] = 1.0
        else:
            raise RuntimeError('Unknown mat type.')
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_diffuse_albedo(jsd_dict, **kwargs):
    jsd_dict = copy.deepcopy(jsd_dict)
    new_mat_jsd = {}
    for mat_name, mat_jsd in jsd_dict['materials'].items():
        if mat_jsd['type'] == 'svbrdf_inline':
            new_mat_jsd[mat_name] = dict(type='basic_texture',
                                         texture=mat_jsd['diffuse_map'])
        elif mat_jsd['type'] == 'beckmann_inline':
            new_mat_jsd[mat_name] = dict(
                type='basic_texture', texture=mat_jsd['params']['diffuse_map'])
        elif mat_jsd['type'] == 'blinn_phong':
            new_mat_jsd[mat_name] = dict(type='basic',
                                         color=mat_jsd['diffuse'])
        else:
            logger.error("Diffuse albedo does not exist!")
            new_mat_jsd[mat_name] = dict(type='basic',
                                         color=(1.0, 0.0, 1.0))
    jsd_dict['materials'] = new_mat_jsd
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_specular_albedo(jsd_dict, **kwargs):
    jsd_dict = copy.deepcopy(jsd_dict)
    new_mat_jsd = {}
    for mat_name, mat_jsd in jsd_dict['materials'].items():
        if mat_jsd['type'] == 'svbrdf_inline':
            new_mat_jsd[mat_name] = dict(type='basic_texture',
                                         texture=mat_jsd['specular_map'])
        if mat_jsd['type'] == 'beckmann_inline':
            new_mat_jsd[mat_name] = dict(
                type='basic_texture', texture=mat_jsd['params']['specular_map'])
        elif mat_jsd['type'] == 'blinn_phong':
            new_mat_jsd[mat_name] = dict(type='basic',
                                         color=mat_jsd['specular'])
        else:
            new_mat_jsd[mat_name] = dict(type='basic',
                                         color=(0.0, 0.0, 0.0))
    jsd_dict['materials'] = new_mat_jsd
    with jsd.JSDRenderer(jsd_dict, **kwargs) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_diff_component(jsd_dict):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['specular_map'][:] = 0.0
    with jsd.JSDRenderer(jsd_dict, ssaa=3, gamma=None) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_spec_component(jsd_dict):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['diffuse_map'][:] = 0.0
    with jsd.JSDRenderer(jsd_dict, ssaa=3, gamma=None) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        return r.render_to_image()


def render_jsd(jsd_dict, **rend_opts):
    with jsd.JSDRenderer(jsd_dict, **rend_opts) as renderer:
        image = renderer.render_to_image()
    return image


def make_jsd(mesh, camera, clear_color=(1.0, 1.0, 1.0)):
    camera = copy.deepcopy(camera)
    camera.clear_color = clear_color
    jsd_dict = {
        "camera": camera.tojsd(),
        "lights": [],
        "mesh": jsd.export_mesh_to_jsd(mesh),
        "materials": {key: {'type': 'depth'} for key in mesh.materials}
    }
    return jsd_dict


def render_depth(mesh, camera):
    jsd_dict = make_jsd(mesh, camera, clear_color=(0.0, 0.0, 0.0))
    jsd_dict["materials"] = {key: {'type': 'depth'} for key in mesh.materials}
    image = render_jsd(jsd_dict)[:, :, 0]
    return image


def render_mesh_normals(mesh, camera):
    jsd_dict = make_jsd(mesh, camera, clear_color=(0.0, 0.0, 0.0))
    jsd_dict["materials"] = {key: {'type': 'normal'} for key in mesh.materials}
    image = render_jsd(jsd_dict)
    return image


def render_tangents(mesh, camera):
    jsd_dict = make_jsd(mesh, camera, clear_color=(0.0, 0.0, 0.0))
    jsd_dict["materials"] = {key: {'type': 'tangent'} for key in mesh.materials}
    image = render_jsd(jsd_dict)
    return image


def render_bitangents(mesh, camera):
    jsd_dict = make_jsd(mesh, camera, clear_color=(0.0, 0.0, 0.0))
    jsd_dict["materials"] = {key: {'type': 'bitangent'} for key in mesh.materials}
    image = render_jsd(jsd_dict)
    return image


def render_segments(mesh: Mesh, camera,
                    segment_type='material') -> np.ndarray:
    if segment_type == 'material':
        segments = mesh.materials
    elif segment_type == 'group':
        segments = mesh.group_names
    elif segment_type == 'object':
        segments = mesh.object_names
    else:
        raise RuntimeError("Unknown segment type")

    mesh_jsd = jsd.export_mesh_to_jsd(mesh)
    mesh_jsd["materials"] = segments
    for face in mesh_jsd["faces"]:
        face["material"] = face[segment_type]

    camera = copy.deepcopy(camera)
    camera.clear_color = (0, 0, 0)
    jsd_dict = {
        "camera": camera.tojsd(),
        "mesh": mesh_jsd,
        "lights": [],
        "materials": {
            key: {
                'type': 'basic',
                'color': np.full((3,), i+1, dtype=np.float32).tolist()
            } for i, key in enumerate(segments)}
    }
    image = render_jsd(jsd_dict)
    segment_image = (image - 1).astype(int)[:, :, 0]
    return segment_image


def render_wavefront_mtl(mesh: Mesh, camera,
                         materials: Dict[str, WavefrontMaterial],
                         radmap_path=None,
                         **rend_opts):
    jsd_dict = make_jsd(mesh, camera)
    jsd_dict["materials"] = {
        mtl.name: {
            'type': 'blinn_phong',
            'diffuse': mtl.diffuse_color,
            'specular': mtl.specular_color,
            'roughness': 1.0 / math.sqrt((mtl.specular_exponent + 2) / 2.0),
        } for mtl_name, mtl in materials.items()
    }
    if radmap_path is not None:
        jsd_dict['radiance_map'] = {
            'path': radmap_path
        }
    return render_jsd(jsd_dict, **rend_opts)


def render_preview(mesh: Mesh, camera,
                   radmap_path=None,
                   **rend_opts):
    jsd_dict = make_jsd(mesh, camera)
    jsd_dict["materials"] = {
        mtl_name: {
            'type': 'blinn_phong',
            'diffuse': tuple(c/255/1.2
                             for c in QUAL_COLORS[mtl_id % len(QUAL_COLORS)]),
            'specular': (0.044, 0.044, 0.044),
            'roughness': 0.1,
        } for mtl_id, mtl_name in enumerate(mesh.materials)
    }
    if radmap_path is not None:
        jsd_dict['radiance_map'] = {
            'path': radmap_path
        }
    return render_jsd(jsd_dict, **rend_opts)


def render_world_coords(mesh, camera):
    jsd_dict = make_jsd(mesh, camera)
    jsd_dict["materials"] = {key: {'type': 'world'} for key in mesh.materials}
    return render_jsd(jsd_dict)


def render_median_colors(mesh, image, camera):
    pixel_segment_ids = render_segments(mesh, camera)
    median_colors = images.compute_segment_median_colors(
        image, pixel_segment_ids)
    median_image = np.ones(image.shape)
    for segment_id in range(len(median_colors)):
        mask = pixel_segment_ids == segment_id
        median_image[mask, :] = median_colors[segment_id, :]

    return median_image


def render_uvs(mesh, camera):
    if len(mesh.uvs) == 0:
        raise RuntimeError('Mesh does not have UVs')
    jsd_dict = make_jsd(mesh, camera, clear_color=(0, 0, 0, 0))
    jsd_dict["materials"] = {key: {'type': 'uv'} for key in mesh.materials}
    return render_jsd(jsd_dict)[:, :, :2]
