import os
import argparse
import cv2
from rendkit.pfm import save_pfm_texture


parser = argparse.ArgumentParser()
parser.add_argument(dest='in_path', type=str)
parser.add_argument(dest='out_path', type=str)
args = parser.parse_args()


def main():
    if not os.path.exists(args.in_path):
        print("Input file does not exist.")
        return
    if os.path.exists(args.out_path):
        overwrite = input("Overwrite? y/n")
        if overwrite != 'y':
            return

    im = cv2.imread(args.in_path, -1)
    print("Saving to {}".format(args.out_path))
    save_pfm_texture(args.out_path, im)


if __name__ == '__main__':
    main()
