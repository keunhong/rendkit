from typing import Tuple

import logging
import numpy as np
from numpy import linalg
from vispy import app, gloo
from vispy.gloo import gl

from rendkit import vector_utils
from rendkit.camera import BaseCamera
from rendkit.core import Scene
from rendkit import postprocessing as pp


logger = logging.getLogger(__name__)


class _nop():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class BaseRenderer(app.Canvas):
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

        self.size = size

        # Buffer shapes are HxW, not WxH...
        self._rendertex = gloo.Texture2D(
            shape=(size[1], size[0]) + (4,),
            internalformat='rgba32f')
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(
            shape=(size[1], size[0])))

    def set_program(self, vertex_shader, fragment_shader):
        self.active_program = gloo.Program(vertex_shader, fragment_shader)

    def draw(self, camera=None):
        """
        Override and implement drawing logic here. e.g. gloo.clear_color
        """
        raise NotImplementedError

    def render_to_image(self, camera=None, out_size=None) -> np.ndarray:
        """
        Renders to an image.
        :return: image of rendered scene.
        """
        with self._fbo:
            self.draw(camera, out_size)
            pixels: np.ndarray = gloo.util.read_pixels(
                out_type=np.float32, alpha=False)
        return pixels

    def on_resize(self, event):
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.camera.size = self.size
        self.update()

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


class SceneRenderer(BaseRenderer):
    def __init__(self, scene, camera, size=None,
                 gamma=None,
                 ssaa=0,
                 tonemap=None,
                 exposure=1.0,
                 reinhard_thres=3.0,
                 conservative_raster=False,
                 *args, **kwargs):
        if size is None:
            size = camera.size
        super().__init__(size, camera, scene, *args, **kwargs)
        gloo.set_state(depth_test=True)
        if conservative_raster:
            from . import nvidia
            self.conservative_raster = nvidia.conservative_raster(True)
        else:
            self.conservative_raster = _nop()
        self.gamma = gamma
        self.ssaa_scale = min(max(1, ssaa),pp.DownsampleProgram.MAX_SCALE)

        # Initialize main camera.
        self.render_size = (self.size[0] * self.ssaa_scale,
                            self.size[1] * self.ssaa_scale)
        self.rend_target_by_cam = {
            self.camera: pp.create_rend_target(self.render_size)
        }
        logger.info("Render size: {} --SSAAx{}--> {}".format(
            self.size, self.ssaa_scale, self.render_size))

        self.pp_pipeline = pp.PostprocessPipeline(self.size)

        if self.ssaa_scale >= 2:
            self.pp_pipeline.add_program(pp.DownsampleProgram(ssaa))

        if tonemap == 'reinhard':
            logger.info("Tonemapping mode reinhard with threshold={}"
                        .format(reinhard_thres))
            self.pp_pipeline.add_program(pp.ReinhardProgram(reinhard_thres))
        elif tonemap == 'exposure':
            logger.info("Tonemapping mode exposure with exposure={}"
                        .format(exposure))
            self.pp_pipeline.add_program(pp.ExposureProgram(exposure))

        if gamma is not None:
            logger.info("Gamma correction with gamma={}"
                        .format(gamma))
            self.pp_pipeline.add_program(pp.GammaCorrectionProgram(gamma))
        else:
            self.pp_pipeline.add_program(pp.IdentityProgram())

    def draw_scene(self, camera):
        rend_fb, rend_colortex, _ = self.rend_target_by_cam[camera]

        gloo.clear(color=camera.clear_color)
        gloo.set_state(depth_test=True)
        gloo.set_viewport(0, 0, rend_colortex.shape[1], rend_colortex.shape[0])
        for renderable in self.scene.renderables:
            program = renderable.activate(self.scene, camera)
            with self.conservative_raster:
                program.draw(gl.GL_TRIANGLES)

    def draw(self, camera=None, out_size=None):
        if camera is None:
            camera = self.camera

        if out_size is None:
            out_size = (self.physical_size
                        if camera == self.camera else camera.size)

        if camera not in self.rend_target_by_cam:
            rend_fb, rend_colortex, rend_depthtex = pp.create_rend_target(
                tuple(s * self.ssaa_scale for s in out_size))
            self.rend_target_by_cam[camera] = (rend_fb, rend_colortex)

        rend_fb, rend_colortex, rend_depthtex = self.rend_target_by_cam[camera]

        with rend_fb:
            self.draw_scene(camera)

        self.pp_pipeline.draw(rend_colortex, rend_depthtex, out_size)
