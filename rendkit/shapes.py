import random
import numpy as np
from numpy import linalg
from scipy.spatial import Delaunay
from rendkit import jsd
from rendkit.vector_utils import normalized


def make_plane(height, width, material_name='plane'):
    return jsd.import_jsd_mesh({
        "scale": 1,
        "type": "inline",
        "vertices": [
            [width/2, 0.0, -height/2],
            [width/2, 0.0, height/2],
            [-width/2, 0.0, height/2],
            [-width/2, 0.0, -height/2]
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
        "materials": [material_name],
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
    })


def make_random(size=100, material_name='plane'):
    t = np.linspace(0, np.pi, size)
    sig1 = (np.exp(-t * random.uniform(-1, 1))
            * np.cos(t * random.gauss(0, 1)))
    sig2 = (np.exp(-t * random.uniform(-1, 1))
            * np.cos(t * random.gauss(0, 1)))
    y = np.outer(sig1, sig2)
    y /= np.abs(y).max()
    y -= y.min()
    xx = np.linspace(-1, 1, 100)
    x, z = np.meshgrid(xx, xx)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    vertices = np.vstack((x, y, z)).T
    tri = Delaunay(vertices[:, [0, 2]])
    faces = []
    normals = np.zeros(vertices.shape)
    uvs = np.vstack((x, z)).T / 2.0 + 1.0
    for inds in tri.simplices:
        p1 = vertices[inds[0]]
        p2 = vertices[inds[1]]
        p3 = vertices[inds[2]]
        u = normalized(p1 - p2)
        v = normalized(p3 - p2)
        for i in inds:
            n = np.cross(u, v)
            # if np.dot(n, [0, 1, 0]) < 0:
            #     n *= -1
            normals[i] += n
        faces.append({
            "vertices": inds,
            "normals": inds,
            "uvs": inds,
            "material": 0,
        })
    vertices[:, [0, 2]] *= size/2
    normals = normals / linalg.norm(normals, axis=1)[:, None]
    return jsd.import_jsd_mesh({
        "scale": 1,
        "type": "inline",
        "vertices": vertices,
        "uvs": uvs,
        "normals": normals,
        "materials": [material_name],
        "faces": faces})
