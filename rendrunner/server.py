import io
import logging
from xmlrpc.server import SimpleXMLRPCServer

import numpy as np
from vispy import app

from rendkit.jsd import JSDRenderer
from rendrunner import config

logger = logging.getLogger(__name__)

app.use_app('pyglet')
root_canvas = app.Canvas(show=False)


def render(jsd_dict):
    with JSDRenderer(jsd_dict=jsd_dict, shared=root_canvas) as renderer:
        image = renderer.render_to_image()
    logger.info('Rendered image with shape {}'.format(image.shape))
    output = io.BytesIO()
    np.savez(output, image=image)
    return output.getvalue()


if __name__=='__main__':
    console = logging.StreamHandler()
    formatter = logging.Formatter(config.LOG_FORMAT)
    console.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(console)
    root_logger.addHandler(console)
    root_logger.setLevel(config.LOG_LEVEL)

    server = SimpleXMLRPCServer(("127.0.0.1", 8000))
    logger.info("Running rendering service on port 8000")
    server.register_function(render)
    server.serve_forever()
