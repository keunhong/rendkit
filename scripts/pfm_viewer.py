import os
import argparse
import numpy as np
from scipy import misc
from matplotlib import pyplot as plt
from rendkit import pfm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='path', type=str)
    parser.add_argument('-o', '--out', dest='out_path', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("File does not exist.")
        return

    im = pfm.load_pfm_texture(args.path)
    print("min={}, max={}, median={}"
          .format(im.min(axis=(0,1)), im.max(axis=(0,1)), np.median(im, axis=(0,1))))

    plt.imshow(im)
    plt.show()

    if args.out_path:
        print("Saving image to {}".format(args.out_path))
        misc.imsave(args.out_path, im)


if __name__ == '__main__':
    main()
