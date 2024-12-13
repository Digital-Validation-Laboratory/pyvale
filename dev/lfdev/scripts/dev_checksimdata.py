'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np

import mooseherder as mh


def main() -> None:
    data_path = Path('data/examplesims/case01_out_4x3.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    field_key = list(sim_data.node_vars.keys())[0] # type: ignore

    print()
    print('Nodal coords')
    print(sim_data.coords)
    print()
    for ii in range(12):
        print(f'e{ii+1}={sim_data.connect['connect1'][:,ii]}')
    print(sim_data.coords.shape)
    print(sim_data.connect['connect1'].shape)


if __name__ == '__main__':
    main()
