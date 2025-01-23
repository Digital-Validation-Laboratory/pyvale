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
    data_path = Path('src/pyvale/simcases/case23_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    dir = Path.cwd() / 'dev/lsdev/blender_files'
    filename = 'case23.blend'
    filepath = dir / filename
    all_files = os.listdir(dir)
    for ff in all_files:
        if filename == ff:
            os.remove(dir / ff)

    filepath = str(filepath)

    scene = BlenderScene()

    part_location = (0, 0, 0)
    angle = np.radians(90)
    part_rotation = (0, 0, angle)

    part, pv_surf, spat_dim, components = scene.add_part(sim_data=sim_data)
    scene.set_part_location(part, part_location)
    scene.set_part_rotation(part, part_rotation)

    mat_data = MaterialData()
    image_path = '/home/lorna/speckle_generator/images/blender_image_texture_rad2.tiff'
    mat = scene.add_material(mat_data, part, image_path)

    sensor_px = (2464, 2056)
    cam_position = (0, 0, 600)
    focal_length = 25.0
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
    image_path = Path.cwd() / 'dev/lsdev/rendered_images/RBM_x'
    output_path = image_path / 'RBM-x.txt'

    step = 0.07935 / 10
    x_lims = [0, 0.07935]
    rigidbodymotion = RigidBodyMotion(part, step, part_location, image_path, output_path, cam_data)
    rigidbodymotion.rigid_body_motion_x(x_lims, part)

if __name__ == "__main__":
    main()