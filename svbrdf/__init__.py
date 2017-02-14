import logging
import os

from rendkit import pfm
from toolbox import images

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
                 alpha=None,
                 suppress_outliers=True,
                 transposed=False):
        if path is not None:
            if not os.path.exists(path):
                raise FileNotFoundError('The path {} does not exist'.format(path))

            logger.info('Loading path={}'.format(path))

            self.path = path

            data_path = os.path.join(self.path, 'out/reverse')

            with open(os.path.join(data_path, MAP_PARAMS_FNAME), 'r') as f:
                line = f.readline()
                self.alpha, _ = [float(i) for i in line.split(' ')]

            self.diffuse_map = pfm.load_pfm_texture(
                os.path.join(data_path, MAP_DIFF_FNAME), transposed=transposed)

            self.specular_map = pfm.load_pfm_texture(
                os.path.join(data_path, MAP_SPEC_FNAME), transposed=transposed)
            self.normal_map = pfm.load_pfm_texture(
                os.path.join(data_path, MAP_NORMAL_FNAME), transposed=transposed)
            self.spec_shape_map = pfm.load_pfm_texture(
                os.path.join(data_path, MAP_SPEC_SHAPE_FNAME), transposed=transposed)

            logger.info('Loaded shape={}, alpha={}'
                .format(self.diffuse_map.shape, self.alpha))
        else:
            self.diffuse_map = diffuse_map
            self.specular_map = specular_map
            self.spec_shape_map = spec_shape_map
            self.normal_map = normal_map
            self.alpha = alpha

        if suppress_outliers:
            logger.info("Suppressing outliers in diffuse and specular maps.")
            self.diffuse_map = images.suppress_outliers(self.diffuse_map)
            self.specular_map = images.suppress_outliers(self.specular_map)

    @property
    def name(self):
        return os.path.split(os.path.realpath(self.path))[1]

    def save(self, path):
        reverse_path = os.path.join(path, 'out', 'reverse')
        if not os.path.exists(reverse_path):
            os.makedirs(reverse_path)
        pfm.save_pfm_texture(os.path.join(reverse_path, MAP_DIFF_FNAME),
                             self.diffuse_map)
        pfm.save_pfm_texture(os.path.join(reverse_path, MAP_SPEC_FNAME),
                             self.specular_map)
        pfm.save_pfm_texture(os.path.join(reverse_path, MAP_SPEC_SHAPE_FNAME),
                             self.spec_shape_map)
        pfm.save_pfm_texture(os.path.join(reverse_path, MAP_NORMAL_FNAME),
                             self.normal_map)
        with open(os.path.join(reverse_path, MAP_PARAMS_FNAME), 'w') as f:
            f.write("{} {}".format(self.alpha, 0.0))

    def to_jsd(self):
        return dict(
            type='svbrdf_inline',
            diffuse_map=self.diffuse_map,
            specular_map=self.specular_map,
            spec_shape_map=self.spec_shape_map,
            normal_map=self.normal_map,
            alpha=self.alpha)
