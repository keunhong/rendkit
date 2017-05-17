from vispy.gloo import gl
from vispy import app
gl.use_gl('glplus')


def init_headless():
    app.use_app('glfw')
