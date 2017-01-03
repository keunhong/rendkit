import numpy as np
from PIL import Image, ImageDraw

from meshkit import Mesh


def visualize_uv_surface(mesh: Mesh, canvas_size=(1000, 1000)):
    visualizations = []
    for object_id, object_name in enumerate(mesh.object_names):
        image = Image.new(mode='RGBA', size=canvas_size, color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        material_faces = mesh.get_faces(filter={'object': object_id})
        for face in material_faces:
            points = [tuple((canvas_size[0] * mesh.uvs[i, :]).tolist())
                      for i in face['uvs']]
            draw.polygon(
                points, fill=(255, 255, 255, 255), outline=(255, 255, 255, 255))
        del draw
        visualizations.append(np.array(image)[:, :, -1] > 0)

    return visualizations
