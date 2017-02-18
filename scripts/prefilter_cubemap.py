import os
import numpy as np
from vispy import app
from rendkit import cubemap as cm
from matplotlib import pyplot as plt


_package_dir = os.path.dirname(os.path.realpath(__file__))
_resource_dir = os.path.join(_package_dir, '..', 'resources')
_cubemap_dir = os.path.join(_resource_dir, 'cubemaps')

app.use_app('glfw')


def main():
    app.Canvas(show=False)
    cube_faces = cm.unstack_cross(cm.stack_cross(
        cm.load_cube_faces(os.path.join(_cubemap_dir, 'yokohama'))))
    irradiance_map = cm.prefilter_irradiance(cube_faces)
    plt.imshow(np.vstack((
        cm.stack_cross(cube_faces),
        cm.stack_cross(irradiance_map))))
    plt.show()


if __name__ == '__main__':
    main()
