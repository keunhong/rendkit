from rendkit.camera import PerspectiveCamera


def render_video(scene):
    camera = PerspectiveCamera(
        size=(1600, 900), fov=90, near=1, far=1000.0,
        position=[0, 50, -50],
        lookat=(0.0, 0.0, -0.0),
        up=(0.0, 1.0, 0.0),
        clear_color=(1, 1, 1))
    renderer= MyJSDRenderer(jsd, camera, dpi=500,
