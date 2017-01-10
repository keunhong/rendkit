import numpy as np
from numpy import linalg
from vispy import util
from vispy.util.quaternion import Quaternion

from . import graphics_utils
from . import vector_utils


class BaseCamera:
    def __init__(self, size, near, far, clear_color=(1.0, 1.0, 1.0)):
        self.size = size
        self.near = near
        self.far = far
        self.clear_color = clear_color

    @property
    def left(self):
        return -self.size[0] / 2

    @property
    def right(self):
        return self.size[0] / 2

    @property
    def top(self):
        return self.size[1] / 2

    @property
    def bottom(self):
        return -self.size[1] / 2

    def projection_mat(self):
        raise NotImplementedError

    def view_mat(self):
        raise NotImplementedError

    def handle_mouse(self):
        pass

    def unproject(self, x: np.ndarray, y: np.ndarray, depth: np.ndarray):
        return graphics_utils.unproject(*self.size,
                                        self.projection_mat(),
                                        self.view_mat(),
                                        x, y, depth, self.near, self.far)

    def apply_projection(self, points):
        homo = graphics_utils.euclidean_to_homogeneous(points)
        proj = self.projection_mat().dot(self.view_mat().dot(homo.T)).T
        proj = graphics_utils.homogeneous_to_euclidean(proj)[:, :2]
        print(proj)
        proj = (proj + 1) / 2
        proj[:, 0] = (proj[:, 0] * self.size[0])
        proj[:, 1] = self.size[1] - (proj[:, 1] * self.size[1])
        return np.fliplr(proj)

    def get_position(self):
        return linalg.inv(self.view_mat())[:3, 3]

    def tojsd(self):
        raise NotImplementedError()


class CalibratedCamera(BaseCamera):
    def __init__(self, extrinsic: np.ndarray, intrinsic: np.ndarray,
                 size, near, far, *args, **kwargs):
        super().__init__(size, near, far, *args, **kwargs)
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic

    def projection_mat(self):
        return graphics_utils.intrinsic_to_opengl_projection(
            self.intrinsic,
            self.left, self.right, self.top, self.bottom,
            self.near, self.far)

    def view_mat(self):
        return graphics_utils.extrinsic_to_opengl_modelview(self.extrinsic)

    def tojsd(self):
        return {
            'type': 'calibrated',
            'size': self.size,
            'near': float(self.near),
            'far': float(self.far),
            'extrinsic': self.extrinsic.tolist(),
            'intrinsic': self.intrinsic.tolist(),
            'clear_color': self.clear_color,
        }


class PerspectiveCamera(BaseCamera):

    def __init__(self, size, near, far, fov, position, lookat, up,
                 *args, **kwargs):
        super().__init__(size, near, far, *args, **kwargs)

        self.fov = fov
        self.position = np.array(position, dtype=np.float32)
        self.lookat = np.array(lookat, dtype=np.float32)
        self.up = vector_utils.normalized(np.array(up))

    def forward(self):
        return vector_utils.normalized(self.lookat - self.position)

    def projection_mat(self):
        mat = util.transforms.perspective(
            self.fov, self.size[0] / self.size[1], self.near, self.far).T
        return mat

    def view_mat(self):
        forward = self.forward()
        rotation_mat = np.eye(3)
        rotation_mat[0, :] = vector_utils.normalized(
            np.cross(forward, self.up))
        rotation_mat[2, :] = -forward
        # We recompute the 'up' vector portion of the matrix as the cross
        # product of the forward and sideways vector so that we have an ortho-
        # normal basis.
        rotation_mat[1, :] = np.cross(rotation_mat[2, :], rotation_mat[0, :])

        position = rotation_mat.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position

        return view_mat


def _get_arcball_vector(x, y, w, h, r=100.0):
    P = np.array((2.0 * x / w - 1.0,
                  -(2.0 * y / h - 1.0),
                  0))
    OP_sq = P[0] ** 2 + P[1] ** 2
    if OP_sq <= 1:
        P[2] = np.sqrt(1 - OP_sq)
    else:
        P = vector_utils.normalized(P)

    return P


class ArcballCamera(PerspectiveCamera):

    def __init__(self, size, near, far, fov, position, lookat, up,
                 rotate_speed=100.0, *args, **kwargs):
        super().__init__(size, near, far, fov, position, lookat, up,
                         *args, **kwargs)
        self.rotate_speed = rotate_speed
        self.max_speed = np.pi / 2

    @classmethod
    def from_perspective(cls, pc: PerspectiveCamera):
        return cls(size=pc.size, near=pc.near, far=pc.far, fov=pc.fov,
                   position=pc.position, lookat=pc.lookat, up=pc.up)

    def handle_mouse(self, last_pos, cur_pos):
        va = _get_arcball_vector(*cur_pos, *self.size)
        vb = _get_arcball_vector(*last_pos, *self.size)
        angle = min(np.arccos(min(1.0, np.dot(va, vb))) * self.rotate_speed,
                    self.max_speed)
        axis_in_camera_coord = np.cross(va, vb)

        cam_to_world = self.view_mat()[:3, :3].T
        axis_in_world_coord = cam_to_world.dot(axis_in_camera_coord)

        rotation_quat = Quaternion.create_from_axis_angle(angle,
                                                          *axis_in_world_coord)
        self.position = rotation_quat.rotate_point(self.position)