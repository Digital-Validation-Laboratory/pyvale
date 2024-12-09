"""Example to create a scene and render a set of images of object undergoing
    in plane rigid body motion
"""

import os
from pathlib import Path
import mooseherder as mh
from dev_blenderscene import BlenderScene
from dev_partblender import *
from dev_objectmaterial import MaterialData
from dev_blendercamera import CameraData
from dev_lightingblender import LightData, LightType
from dev_rigidbodymotion import RigidBodyMotion


def main() -> None:
    # Making Blender scene
    data_path = Path('src/data/case13_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    dir = Path.cwd() / 'dev/lsdev/blender_files'
    filename = 'case13.blend'
    filepath = dir / filename
    all_files = os.listdir(dir)
    for ff in all_files:
        if filename == ff:
            os.remove(dir / ff)

    filepath = str(filepath)

    scene = BlenderScene()

    part_location = (0, 0, 0)
    part = scene.add_part(sim_data)
    scene.set_part_location(part, part_location)

    mat_data = MaterialData()
    image_path = '/home/lorna/speckle_generator/images/blender_image_texture_rad2.tiff'
    mat = scene.add_material(mat_data, part, image_path)

    sensor_px = (2464, 2056)
    cam_position = (0, 0, 200)
    focal_length = 15.0
    cam_data = CameraData(sensor_px=sensor_px,
                          position=cam_position,
                          focal_length=focal_length)

    camera = scene.add_camera(cam_data)

    type = LightType.POINT
    light_position = (0, 0, 200)
    energy = 500 * (10)**3
    light_data = LightData(type=type,
                           position=light_position,
                           energy=energy)

    light = scene.add_light(light_data)

    #---------------------------------------------------------------------------
    # Rendering images
    image_path = Path.cwd() / 'dev/lsdev/rendered_images/RBM_1-5mm'
    output_path = Path.cwd() / 'dev/lsdev/rendered_images'

    step = 1
    x_lims = [0, 5]
    rigidbodymotion = RigidBodyMotion(part, step, part_location, image_path, output_path, cam_data)
    rigidbodymotion.rigid_body_motion_x(x_lims)

if __name__ == "__main__":
    main()