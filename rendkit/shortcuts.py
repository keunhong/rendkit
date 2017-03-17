import copy

import numpy as np

from rendkit import jsd
from .camera import ArcballCamera
from .jsd import JSDRenderer


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


def render_full(jsd_dict, uv_scale=6.0):
    with jsd.JSDRenderer(jsd_dict, ssaa=3, gamma=None) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        for renderable in r.scene.renderables:
            renderable.scale_uvs(uv_scale)
        return r.render_to_image()


def render_lambert_map(jsd_dict, uv_scale=6.0):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['diffuse_map'][:] = 1.0
            mat_jsd['specular_map'][:] = 0.0
    with jsd.JSDRenderer(jsd_dict, ssaa=3, gamma=None) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        for renderable in r.scene.renderables:
            renderable.scale_uvs(uv_scale)
        return r.render_to_image()


def render_diff_component(jsd_dict, uv_scale=6.0):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['specular_map'][:] = 0.0
    with jsd.JSDRenderer(jsd_dict, ssaa=3, gamma=None) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        for renderable in r.scene.renderables:
            renderable.scale_uvs(uv_scale)
        return r.render_to_image()


def render_spec_component(jsd_dict, uv_scale=6.0):
    jsd_dict = copy.deepcopy(jsd_dict)
    for mat_name in jsd_dict['materials']:
        mat_jsd = jsd_dict['materials'][mat_name]
        if mat_jsd['type'] == 'svbrdf_inline':
            mat_jsd['diffuse_map'][:] = 0.0
    with jsd.JSDRenderer(jsd_dict, ssaa=3, gamma=None) as r:
        r.camera.clear_color = (0.0, 0.0, 0.0)
        for renderable in r.scene.renderables:
            renderable.scale_uvs(uv_scale)
        return r.render_to_image()
