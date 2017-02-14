import copy

import numpy as np

from .camera import ArcballCamera
from .jsd import JSDRenderer


def svbrdf_plane_renderer(svbrdf, size=None, lights=list(), radmap=None,
                          mode='all', gamma=2.2, uv_scale=1.0, shape=None,
                          transpose=False, cam_lookat=(0.0, 0.0), cam_fov=90,
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

    camera = ArcballCamera(
        size=size, fov=cam_fov, near=0.1, far=1000.0,
        position=[0, cam_dist, 0],
        lookat=(cam_lookat[0], 0.0, -cam_lookat[1]), up=cam_up)

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


def radmap_scale_plane(size=50):
    jsd = {
        "mesh": {
            "scale": 1,
            "type": "inline",
            "vertices": [
                [size, 0.0, -size],
                [size, 0.0, size],
                [-size, 0.0, size],
                [-size, 0.0, -size]
            ],
            "uvs": [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
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
