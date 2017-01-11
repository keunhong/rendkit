import argparse
import json
import logging

import numpy as np
from vispy import app

from rendkit.camera import ArcballCamera
from rendkit import jsd


LOG_FORMAT = '%(asctime)s\t%(levelname)s\t%(message)s\t[%(name)s]'


app.use_app('pyglet')

parser = argparse.ArgumentParser()
parser.add_argument('--jsd', dest='jsd_path', type=str, required=True)
parser.add_argument('--ssaa', dest='ssaa', type=int, default=2)
parser.add_argument('--gamma', dest='gamma', type=float, default=None)

args = parser.parse_args()

np.set_printoptions(suppress=True)


class MyJSDRenderer(jsd.JSDRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, show=True)
        self.uv_scale = 1.0

    def on_key_press(self, event):
        if event.key == 'Escape':
            self.app.quit()
            return

        old_uv_scale = self.uv_scale
        if event.key == '=':
            self.update_uv_scale(self.uv_scale * 2)
            print('(+) UV scale {} -> {}'.format(old_uv_scale,
                                                 self.uv_scale))
        elif event.key == '-':
            self.update_uv_scale(self.uv_scale / 2)
            print('(-) UV scale {} -> {}'.format(old_uv_scale,
                                                 self.uv_scale))
        self.draw()

    def update_uv_scale(self, scale):
        for renderable in self.scene.renderables:
            renderable.scale_uvs(scale)
        self.uv_scale = scale
        self.update()


if __name__ == '__main__':
    console = logging.StreamHandler()
    formatter = logging.Formatter(LOG_FORMAT)
    console.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(console)
    root_logger.addHandler(console)
    root_logger.setLevel(logging.INFO)
    with open(args.jsd_path, 'r') as f:
        jsd = json.load(f)

    camera = ArcballCamera(
        size=(1600, 900), fov=75, near=1, far=1000.0,
        position=[0, 100, 100],
        lookat=(0.0, 0.0, -0.0),
        up=(0.0, 1.0, 0.0))
    renderer= MyJSDRenderer(
        jsd, camera, dpi=500, gamma=args.gamma, ssaa=args.ssaa)
    app.run()
