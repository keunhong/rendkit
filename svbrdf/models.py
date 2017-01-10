import pickle

from scipy.misc import imread
from sqlalchemy import (Column, Integer, String, Float, LargeBinary,
                         ForeignKey)
from sqlalchemy.orm import relationship

from gravel.database import Base


class SVBRDF(Base):
    __tablename__ = 'svbrdfs'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    substance_name = Column(String(100), nullable=True)
    path = Column(String(1024))

    def get_jsd(self):
        return {
            "type": 'svbrdf',
            "path": self.path,
        }


class SVBRDFRendering(Base):
    __tablename__ = 'svbrdf_renderings'

    id = Column(Integer, primary_key=True)
    svbrdf_id = Column(Integer, ForeignKey(SVBRDF.id))
    image_path = Column(String(1024))

    color_mean_l = Column(Float)
    color_mean_a = Column(Float)
    color_mean_b = Column(Float)
    color_std_l = Column(Float)
    color_std_a = Column(Float)
    color_std_b = Column(Float)
    uv_scale = Column(Float)

    _lights = Column(LargeBinary, nullable=False)

    svbrdf = relationship('SVBRDF',
                         foreign_keys='SVBRDFRendering.svbrdf_id')

    @property
    def lights(self):
        if self._lights:
            return pickle.loads(self._lights)
        return None

    @lights.setter
    def lights(self, lights):
        self._lights = pickle.dumps(lights)

    def load_image(self, rescale_to_single=True):
        image = imread(self.image_path)
        if rescale_to_single:
            return image / 255.0
        return image

    @property
    def feature(self):
        return SVBRDFRenderingFeature.query \
            .filter_by(rendering_id=self.id) \
            .first()


class SVBRDFRenderingFeature(Base):
    __tablename__ = 'svbrdf_rendering_features'

    id = Column(Integer, primary_key=True)
    rendering_id = Column(Integer, ForeignKey(SVBRDFRendering.id))

    patch_vis_path = Column(String(1024), nullable=True)
    _feat_dicts = Column(LargeBinary, nullable=False)
    _patch_scales = Column(LargeBinary, nullable=False)

    rendering = relationship(
        'SVBRDFRendering',
        foreign_keys='SVBRDFRenderingFeature.rendering_id')

    @property
    def feat_dicts(self):
        if self._feat_dicts:
            return pickle.loads(self._feat_dicts)

    @feat_dicts.setter
    def feat_dicts(self, feat_dicts):
        self._feat_dicts = pickle.dumps(feat_dicts)

    @property
    def patch_scales(self):
        if self._patch_scales:
            return pickle.loads(self._patch_scales)

    @patch_scales.setter
    def patch_scales(self, patch_scales):
        self._patch_scales = pickle.dumps(patch_scales)
