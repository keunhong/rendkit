import os
import sys
import tempfile

import bpy
import json

import math

from meshkit import wavefront
from rendkit import jsd
import argparse

from toolbox.logging import init_logger

logger = init_logger(__name__)


def reset_blender():
    bpy.ops.wm.read_factory_settings()

    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)

    # only worry about data in the startup scene
    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras,
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)


def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def assign_material(jsd_mat, bpy_mat):
    bpy_mat.use_nodes = True
    nodes = bpy_mat.node_tree.nodes
    diff_brdf_node = nodes.new(type='ShaderNodeBsdfDiffuse')
    diff_brdf_node.color = (1.0, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='jsd_path', type=str)
    parser.add_argument(dest='out_path', type=str)
    args = parser.parse_args()

    with open(args.jsd_path, 'r') as f:
        jsd_dict = json.load(f)

    reset_blender()

    logger.info("Importing mesh {}".format(jsd_dict['mesh']['path']))
    mesh = wavefront.read_obj_file(jsd_dict['mesh']['path'])
    mesh.resize(1)
    mesh_tmp_path = '/tmp/_jsd_to_cycles.tmp.obj'
    wavefront.save_obj_file(mesh_tmp_path, mesh)

    bpy.ops.import_scene.obj(filepath=mesh_tmp_path,
                             use_edges=False, use_smooth_groups=False,
                             use_split_objects=False, use_split_groups=False,
                             use_groups_as_vgroups=False,
                             use_image_search=True)
    # bpy.context.object.location = (0, 0, 0)

    bpy.ops.object.camera_add()
    scene = bpy.context.scene
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.camera = bpy.context.object
    scene.camera.location = (0.877, 1.556, 0.874)
    scene.camera.rotation_euler = (1.109, 0, 2.617)
    if scene.render.engine != 'CYCLES':
        logger.info("Setting renderer engine {} -> CYCLES"
                    .format(scene.render.engine))
        scene.render.engine = 'CYCLES'

    for bpy_mat in bpy.data.materials:
        mat_name = bpy_mat.name
        if mat_name in jsd_dict['materials']:
            logger.info("Processing material {}".format(mat_name))
            assign_material(jsd_dict['materials'][mat_name], bpy_mat)

    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.user_preferences.addons['cycles'].preferences\
        .compute_device_type = 'CUDA'
    bpy.context.user_preferences.addons['cycles'].preferences\
        .devices[0].use = True
    bpy.data.scenes['Scene'].render.filepath = '/home/kpar/test.png'
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
