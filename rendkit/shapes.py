from rendkit import jsd


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
