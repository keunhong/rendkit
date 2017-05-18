import math
import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.pyplot import tight_layout
from tqdm import trange, tqdm

from rendkit.camera import PerspectiveCamera
from rendkit.renderers import SceneRenderer
from toolbox.logging import init_logger

logger = init_logger(__name__)


def render_frames(scene, size, n_frames=240):
    camera = PerspectiveCamera(
        size=size, fov=75, near=0.1, far=1000.0,
        position=[0, 0, 0], lookat=(0.0, 0.0, -0.0),
        up=(0.0, 1.0, 0.0),
        clear_color=(1, 1, 1))
    frames = []
    with SceneRenderer(scene, camera, tonemap='reinhard', reinhard_thres=3.0,
                       gamma=2.2, ssaa=3) as r:
        for i in trange(n_frames):
            angle = i * 2 * math.pi / (n_frames/2)
            phi = angle/2
            dist = 90 + 20 * math.cos(phi*1.3)
            camera.position[:] = (
                dist * math.cos(angle),
                70 + 10 * (math.log(n_frames) - math.log(i+1)) * math.cos(phi),
                dist * math.sin(angle))
            camera.lookat[:] = (0, 30 * -math.sin(phi), 0)
            frame = np.clip(r.render_to_image(camera), 0, 1)
            frames.append(frame)
    return frames


def save_mp4(path, frames, size, fps=30):
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect('auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')

    im = ax.imshow(frames[0], interpolation='bilinear')
    im.set_clim([0, 1])
    fig.set_size_inches([size[0]/100, size[1]/100])

    with tqdm(total=len(frames)) as pbar:
        def update_im(frame_no):
            pbar.update(1)
            im.set_data(frames[frame_no])
            return im
        ani = animation.FuncAnimation(fig, update_im, len(frames), interval=1)
        writer = animation.writers['ffmpeg'](fps=fps)
        ani.save(str(path), writer=writer, dpi=100)
