import copy

import numpy as np

from .camera import ArcballCamera
from .jsd import JSDRenderer


def svbrdf_plane_renderer(svbrdf, lights=list(), radiance_map=None, mode='all',
                          gamma=2.2, uv_scale=1.0, shape=None, **kwargs):
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
        svbrdf.specular_map = zeros
        svbrdf.diffuse_map = ones

    camera = ArcballCamera(
        size=(width, height), fov=90, near=10, far=1000.0,
        position=[0, min(width, height)/max(width, height)*100, 0],
        lookat=(0.0, 0.0, -0.0),
        up=(0.0, 0.0, -1.0))

    jsd = {
        "mesh": {
            "scale": 100,
            "type": "inline",
            "vertices": [
                [width/height, 0.0, -1.0],
                [width/height, 0.0, 1.0],
                [-width/height, 0.0, 1.0],
                [-width/height, 0.0, -1.0]
            ],
            "uvs": [
                [uv_scale, 0.0],
                [uv_scale, uv_scale],
                [0.0, uv_scale],
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
    if radiance_map is not None:
        jsd['radiance_map'] = radiance_map
    return JSDRenderer(jsd, camera, size=(int(width), int(height)), gamma=gamma,
                       **kwargs)
