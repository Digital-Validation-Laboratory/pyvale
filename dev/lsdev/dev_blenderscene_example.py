import os
from pathlib import Path
import numpy as np
import mooseherder as mh
from blenderscene import BlenderScene
from dev_partblender import *

def main() -> None:
    # cwd = Path.cwd()
    # data_path = cwd.parents[0] / 'mooseherder/scripts/moose/moose-mech-simple_out.e'
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

    camera = scene.add_camera()

    light = scene.add_light()

    scene.save_model(filepath)

if __name__ == "__main__":
    main()