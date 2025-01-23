"""Example to create a scene in Blender and save it as a Blender file
"""

import os
from pathlib import Path
import numpy as np
import mooseherder as mh
from dev_blenderscene import BlenderScene
from dev_partblender import *
from dev_blendercamera import CameraData
from dev_lightingblender import LightData, LightType
from dev_objectmaterial import MaterialData

def main() -> None:
    data_path = Path('src/pyvale/simcases/case24_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    dir = Path.cwd() / 'dev/lsdev/blender_files'
    filename = 'case24.blend'
    filepath = dir / filename
    all_files = os.listdir(dir)
    for ff in all_files:
        if filename == ff:
            os.remove(dir / ff)

    filepath = str(filepath)

    scene = BlenderScene()

    part_location = (0, 0, 0)
    # meshfile = '/home/lorna/pyvale/test_output/test_mesh.stl'

    part, pv_surf, spat_dim, components = scene.add_stl_part(sim_data=sim_data)
    scene.set_part_location(part=part, location=part_location)
    print(f"{part.data.attributes=}")

    mat_data = MaterialData()
    # image_path = str(Path('dev/lsdev/rendered_images/blender_image_texture.tiff'))
    image_path = '/home/lorna/speckle_generator/images/blender_image_texture_rect.tiff'
    mat = scene.add_material(mat_data, part, image_path)

    sensor_px = (2464, 2056)
    cam_position = (0, 0, 150)
    focal_length = 15.0
    cam_data = CameraData(sensor_px=sensor_px,
                          position=cam_position,
                          focal_length=focal_length)

    camera = scene.add_camera(cam_data)

    type = LightType.POINT
    light_position = (0, 0, 200)
    energy = 200 * (10)**3
    light_data = LightData(type=type,
                           position=light_position,
                           energy=energy)

    light = scene.add_light(light_data)

    scene.save_model(filepath)

if __name__ == "__main__":
    main()