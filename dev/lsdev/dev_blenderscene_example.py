import os
from pathlib import Path
import numpy as np
import mooseherder as mh
from blenderscene import BlenderScene
from dev_partblender import *
from camera import CameraData
from lightingblender import LightData, LightType

def main() -> None:
    # cwd = Path.cwd()
    # data_path = cwd.parents[0] / 'mooseherder/scripts/moose/moose-mech-simple_out.e'
    data_path = Path('src/data/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    dir = Path.cwd() / 'dev/lsdev/blender_files'
    filename = 'case17.blend'
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

    sensor_px = (2452, 2056)
    cam_position = (0, 0, 2)
    focal_length = 25.0
    cam_data = CameraData(sensor_px=sensor_px,
                          position=cam_position,
                          focal_length=focal_length)

    camera = scene.add_camera(cam_data)

    type = LightType.SPOT
    light_position = (0, 0, 5)
    energy = 20
    light_data = LightData(type=type,
                           position=light_position,
                           energy=energy)

    light = scene.add_light(light_data)

    scene.save_model(filepath)

if __name__ == "__main__":
    main()