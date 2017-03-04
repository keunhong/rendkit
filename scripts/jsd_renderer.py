import argparse
import json
import logging

import numpy as np
from vispy import app

from rendkit import jsd
from rendkit.camera import ArcballCamera
from rendkit.materials import SVBRDFMaterial

LOG_FORMAT = '%(asctime)s\t%(levelname)s\t%(message)s\t[%(name)s]'


app.use_app('pyglet')

parser = argparse.ArgumentParser()
parser.add_argument('--jsd', dest='jsd_path', type=str, required=True)
parser.add_argument('--ssaa', dest='ssaa', type=int, default=2)
parser.add_argument('--gamma', dest='gamma', type=float, default=None)
parser.add_argument('--exposure', dest='exposure', type=float, default=1.0)

args = parser.parse_args()

np.set_printoptions(suppress=True)


class MyJSDRenderer(jsd.JSDRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, show=True)
        self.uv_scale = 1.0
        self.current_mat_idx = 0
        self.current_mat = None

    def on_key_press(self, event):
        if event.key == 'Escape':
            self.app.quit()
            return

        move_amount = 2.0

        old_uv_scale = self.uv_scale
        if event.key == '=':
            self.update_uv_scale(self.uv_scale * 2)
            print('(+) UV scale {} -> {}'.format(old_uv_scale,
                                                 self.uv_scale))
        elif event.key == '-':
            self.update_uv_scale(self.uv_scale / 2)
            print('(-) UV scale {} -> {}'.format(old_uv_scale,
                                                 self.uv_scale))
        elif event.key == '1':
            mat_names, mats = zip(*list(self.scene.materials.items()))
            self.current_mat_idx  = (self.current_mat_idx + 1) % len(mats)
            self.current_mat = mats[self.current_mat_idx]
            print("Selecting material {}: {}".format(
                self.current_mat_idx,
                mat_names[self.current_mat_idx]))
        elif event.key == 'a':
            if self.current_mat is not None:
                if isinstance(self.current_mat, SVBRDFMaterial):
                    self.current_mat.alpha -= 0.1
                    print('Setting alpha to {}'.format(self.current_mat.alpha))
                    self.recompile_renderables()
        elif event.key == 's':
            if self.current_mat is not None:
                if isinstance(self.current_mat, SVBRDFMaterial):
                    self.current_mat.alpha += 0.1
                    print('Setting alpha to {}'.format(self.current_mat.alpha))
                    self.recompile_renderables()
        elif event.key == 'z':
            if self.current_mat is not None:
                if isinstance(self.current_mat, SVBRDFMaterial):
                    self.current_mat.spec_shape_map[:, :, :] /= 1.1
                    print(np.mean(self.current_mat.spec_shape_map, axis=(0,1)))
                    self.recompile_renderables()
        elif event.key == 'x':
            if self.current_mat is not None:
                if isinstance(self.current_mat, SVBRDFMaterial):
                    self.current_mat.spec_shape_map[:, :, :] *= 1.1
                    print(np.mean(self.current_mat.spec_shape_map, axis=(0,1)))
                    self.recompile_renderables()
        self.update()
        self.draw()

    def recompile_renderables(self):
        for renderable in self.scene.renderables:
            renderable._program = renderable.compile(self.scene)

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
    renderer= MyJSDRenderer(jsd, camera, dpi=500, gamma=args.gamma,
                            ssaa=args.ssaa, exposure=args.exposure)
    app.run()
