import argparse
import os
from scipy.misc import imread

from rendkit.envmap.conversion import panorama_to_cubemap
from rendkit.envmap.io import unstack_cross, load_envmap, stack_cross
from vispy import app
from matplotlib import pyplot as plt


_package_dir = os.path.dirname(os.path.realpath(__file__))
_resource_dir = os.path.join(_package_dir, '..', 'resources')
_cubemap_dir = os.path.join(_resource_dir, 'cubemaps')

app.use_app('glfw')


parser = argparse.ArgumentParser()
parser.add_argument(dest='path', type=str)
parser.add_argument('-o', '--out', dest='out_path', type=str, required=True)
args = parser.parse_args()


def main():
    app.Canvas(show=False)
    panorama = imread(args.path)
    cubemap = panorama_to_cubemap(panorama)
    plt.imshow(stack_cross(cubemap))
    plt.show()


if __name__ == '__main__':
    main()
