from pathlib import Path
import numpy as np
import mooseherder as mh
from blenderscene import BlenderScene
from dev_partblender import *

def main() -> None:
    data_path = Path('src/data/case13_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()


    filepath = '/home/lorna/pyvale/dev/lsdev/blender_files/case13.blend'
    scene = BlenderScene()

    part_location = (0, 0, 0)
    part = scene.add_part(sim_data, location=part_location)


    camera = scene.add_camera()

    scene.save_model(filepath)

if __name__ == "__main__":
    main()