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
    cubemap = cm.load_cubemap(os.path.join(_cubemap_dir, 'yokohama'))
    processor = cm.LambertPrefilterProcessor()
    result = processor.filter(cubemap)
    plt.imshow(np.vstack((
        cm.stack_cross(cubemap),
        cm.stack_cross(result))))
    plt.show()


if __name__ == '__main__':
    main()
