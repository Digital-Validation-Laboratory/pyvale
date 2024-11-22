from pathlib import Path
import numpy as np
import mooseherder as mh
from blenderscene import BlenderScene

def main() -> None:
    data_path = Path('src/data/case13_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    scene = BlenderScene()

    scene.add_part(sim_data)

    filepath = '/home/lorna/pyvale/dev/lsdev/trial_blender_part'
    scene.save_model(filepath)