import io
from xmlrpc.client import ServerProxy
import numpy as np
from scipy.misc import imsave


JSD_TEST = {
    "camera": {
        "type": "perspective",
        "size": (500, 500),
        "fov": 75,
        "near": 0.1,
        "far": 1000.0,
        "position": [0, 0, 150],
        "lookat": (0.0, 0.0, -0.0),
        "up": (0.0, 1.0, 0.0),
    },
    "mesh": {
        "scale": 100,
        "type": "inline",
        "vertices": [
            [1.000000, -1.000000, 0.000000],
            [1.000000, 1.000000, 0.000000],
            [-1.000000, 1.000000, 0.000000],
            [-1.000000, -1.000000, 0.000000]
        ],
        "uvs": [
            [1.000000, 0.000000],
            [1.000000, 1.000000],
            [0.000000, 1.000000],
            [0.000000, 0.000000]
        ],
        "normals": [
            [0, 0, 1]
        ],
        "materials": [
            "plane"
        ],
        "faces": [
            {"vertices": [0, 1, 2], "uvs": [0, 1, 2], "normals": [0, 0, 0], "material": 0},
            {"vertices": [0, 2, 3], "uvs": [0, 2, 3], "normals": [0, 0, 0], "material": 0}
        ]
    },
    "lights": [
        {
            "type": "point",
            "position": [0, 0, 1],
            "intensity": 5000
        },
    ],
    "materials": {
        "plane": {
            "type": "basic",
            "color": [0, 1, 1],
        }
    }
}


if __name__=='__main__':
    proxy = ServerProxy('http://localhost:8000/')
    result = proxy.render(JSD_TEST)
    image = np.load(io.BytesIO(result.data))['image']
    imsave('/Users/kpar/Desktop/test.png', image)
