import json
import argparse
from pathlib import Path

from scipy.misc import imsave

import rendkit
from rendkit import video
from rendkit.jsd import import_jsd_scene
from toolbox.logging import init_logger, disable_logging


rendkit.init_headless()
logger = init_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument(dest='jsd_path', type=str)
parser.add_argument(dest='out_path', type=str)
args = parser.parse_args()


def main():
    disable_logging('svbrdf')

    with open(args.jsd_path, 'r') as f:
        jsd_dict = json.load(f)

    logger.info("Loading scene.")
    scene = import_jsd_scene(jsd_dict, show_floor=True, shadows=True)
    video.save_mp4(args.out_path, video.frame_generator(scene, n_frames=100))


if __name__ == '__main__':
    main()