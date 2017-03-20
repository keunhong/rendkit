import argparse
import os

from rendkit.envmap.conversion import panorama_to_cubemap
from rendkit.envmap.io import unstack_cross, load_envmap, stack_cross
from vispy import app
from rendkit.pfm import pfm_read, pfm_write


_package_dir = os.path.dirname(os.path.realpath(__file__))
_resource_dir = os.path.join(_package_dir, '..', 'resources')
_cubemap_dir = os.path.join(_resource_dir, 'cubemaps')

app.use_app('glfw')


parser = argparse.ArgumentParser()
parser.add_argument(dest='in_path', type=str)
parser.add_argument(dest='out_path', type=str)
args = parser.parse_args()


def main():
    if not os.path.exists(args.in_path):
        print("Input file does not exist.")
        return
    if os.path.exists(args.out_path):
        overwrite = input("Overwrite? y/n: ")
        if overwrite != 'y':
            print("Aborting.")
            return

    panorama = pfm_read(args.in_path, transposed=True)
    cubemap = panorama_to_cubemap(panorama)
    cross = stack_cross(cubemap)
    print("Saving to {}".format(args.out_path))
    pfm_write(args.out_path, cross)


if __name__ == '__main__':
    main()
