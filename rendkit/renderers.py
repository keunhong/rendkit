from typing import Tuple

import logging
import numpy as np
from numpy import linalg

from rendkit import vector_utils
from rendkit.camera import BaseCamera
from rendkit.core import Scene
from vispy import app, gloo


logger = logging.getLogger(__name__)


class _nop():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class Renderer(app.Canvas):
    def __init__(self, size: Tuple[int, int],
                 camera: BaseCamera,
                 scene: Scene,
                 *args, **kwargs):
        if size is None:
            size = camera.size
        self.camera = camera
        self.scene = scene
        super().__init__(size=size, *args, **kwargs)
        gloo.set_state(depth_test=True)
        gloo.set_viewport(0, 0, *self.size)

        self.active_program = None
        self.size = size

        # Buffer shapes are HxW, not WxH...
        self._rendertex = gloo.Texture2D(
            shape=(size[1], size[0]) + (4,),
            internalformat='rgba32f')
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(
            shape=(size[1], size[0])))

        self.mesh = None

        self._current_material = None

    def set_program(self, vertex_shader, fragment_shader):
        self.active_program = gloo.Program(vertex_shader, fragment_shader)

    def draw(self):
        """
        Override and implement drawing logic here. e.g. gloo.clear_color
        """
        raise NotImplementedError

    def render_to_image(self) -> np.ndarray:
        """
        Renders to an image.
        :return: image of rendered scene.
        """
        with self._fbo:
            self.draw()
            pixels = gloo.util.read_pixels(out_type=np.float32,
                                           alpha=False)
        return pixels

    def on_resize(self, event):
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.camera.size = self.size

    def on_draw(self, event):
        self.draw()

    def on_mouse_move(self, event):
        if event.is_dragging:
            self.camera.handle_mouse(event.last_event.pos, event.pos)
            self.update()

    def on_mouse_wheel(self, event):
        cur_dist = linalg.norm(self.camera.position)
        zoom_amount = 5.0 * event.delta[1]
        zoom_dir = vector_utils.normalized(self.camera.position)
        if cur_dist - zoom_amount > 0:
            self.camera.position -= zoom_amount * zoom_dir
            self.update()

    def __enter__(self):
        self._backend._vispy_warmup()
        return self


class DummyRenderer(app.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gloo.set_viewport(0, 0, *self.size)

    def __enter__(self):
        self._backend._vispy_warmup()
        return self


class ContextProvider:
    def __init__(self, size):
        self.size = size
        canvas = gloo.get_current_canvas()
        self.context_exists = canvas is not None and not canvas._closed
        if self.context_exists:
            logger.debug("Using existing OpenGL context.")
            self.provider = gloo.get_current_canvas()
            self.previous_size = self.provider.size
        else:
            logger.debug("Providing temporary context with DummyRenderer.")
            self.provider = DummyRenderer(size=size)

    def __enter__(self):
        gloo.set_viewport(0, 0, *self.size)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.context_exists:
            self.provider.close()
        else:
            gloo.set_viewport(0, 0, *self.previous_size)
