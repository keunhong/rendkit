import math
import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.pyplot import tight_layout

from rendkit.camera import PerspectiveCamera
from rendkit.renderers import SceneRenderer
from toolbox.logging import init_logger

logger = init_logger(__name__)


def frame_generator(scene, size=(800, 800), n_frames=120):
    camera = PerspectiveCamera(
        size=size, fov=90, near=1, far=1000.0,
        position=[0, 0, 0], lookat=(0.0, 0.0, -0.0),
        up=(0.0, 1.0, 0.0),
        clear_color=(1, 1, 1))
    radius = 80
    cam_height = 60
    with SceneRenderer(scene, camera, tonemap='reinhard', reinhard_thres=3.0,
                       gamma=2.2, ssaa=3) as r:
        for i in range(n_frames):
            logger.info("Rendering frame {}".format(i))
            angle = i * 2 * math.pi / n_frames
            camera.position[0] = radius * math.cos(angle)
            camera.position[1] = cam_height
            camera.position[2] = radius * math.sin(angle)
            frame = np.clip(r.render_to_image(camera), 0, 1)
            yield frame


def save_mp4(path, frames):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(next(frames), interpolation='nearest')
    im.set_clim([0, 1])
    fig.set_size_inches([8, 8])
    tight_layout()

    def update_im(x):
        im.set_data(x)
        return im

    tight_layout()

    ani = animation.FuncAnimation(fig, update_im, frames, interval=30)
    writer = animation.writers['ffmpeg'](fps=30)
    ani.save(path, writer=writer, dpi=100)
