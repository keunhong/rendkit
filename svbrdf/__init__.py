import logging
import os

from rendkit import io

MAP_DIFF_FNAME = 'map_diff.pfm'
MAP_SPEC_FNAME = 'map_spec.pfm'
MAP_NORMAL_FNAME = 'map_normal.pfm'
MAP_SPEC_SHAPE_FNAME = 'map_spec_shape.pfm'
MAP_PARAMS_FNAME = 'map_params.dat'


logger = logging.getLogger(__name__)


class SVBRDF:
    def __init__(self, path=None,
                 diffuse_map=None,
                 specular_map=None,
                 spec_shape_map=None,
                 normal_map=None,
                 alpha=None):
        if path is not None:
            if not os.path.exists(path):
                raise FileNotFoundError('The path {} does not exist'.format(path))

            logger.info('[SVBRDF] Loading path={}'.format(path))

            self.path = path

            data_path = os.path.join(self.path, 'out/reverse')

            with open(os.path.join(data_path, MAP_PARAMS_FNAME), 'r') as f:
                line = f.readline()
                self.alpha, _ = [float(i) for i in line.split(' ')]

            self.diffuse_map = io.load_pfm_texture(
                os.path.join(data_path, MAP_DIFF_FNAME))

            self.specular_map = io.load_pfm_texture(
                os.path.join(data_path, MAP_SPEC_FNAME))
            self.normal_map = io.load_pfm_texture(
                os.path.join(data_path, MAP_NORMAL_FNAME))
            self.spec_shape_map = io.load_pfm_texture(
                os.path.join(data_path, MAP_SPEC_SHAPE_FNAME))

            logger.info('[SVBRDF] Loaded size=({}x{}), alpha={}'.format(
                self.diffuse_map.shape[1], self.diffuse_map.shape[0], self.alpha))
        else:
            self.diffuse_map = diffuse_map
            self.specular_map = specular_map
            self.spec_shape_map = spec_shape_map
            self.normal_map = normal_map
            self.alpha = alpha

    def save(self, path):
        reverse_path = os.path.join(path, 'out', 'reverse')
        if not os.path.exists(reverse_path):
            os.makedirs(reverse_path)
        io.save_pfm_texture(os.path.join(reverse_path, MAP_DIFF_FNAME),
                            self.diffuse_map)
        io.save_pfm_texture(os.path.join(reverse_path, MAP_SPEC_FNAME),
                            self.specular_map)
        io.save_pfm_texture(os.path.join(reverse_path, MAP_SPEC_SHAPE_FNAME),
                            self.spec_shape_map)
        io.save_pfm_texture(os.path.join(reverse_path, MAP_NORMAL_FNAME),
                            self.normal_map)
        with open(os.path.join(reverse_path, MAP_PARAMS_FNAME), 'w') as f:
            f.write("{} {}".format(self.alpha, 0.0))


