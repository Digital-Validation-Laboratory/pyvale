'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
from typing import Any

import pyvista as pv

import mooseherder as mh
import pyvale

class VectorField:
    pass

def print_attrs(in_obj: Any) -> None:
    _ = [print(aa) for aa in dir(in_obj) if '__' not in aa]

def main() -> None:
    case_str = 'case17'
    data_path = Path('simcases/'+case_str+'/'+case_str+'_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()


    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    components = ('disp_x','disp_y')    # Same as in the moose input and SimData node_var key
    vec_field = pyvale.Field(sim_data,components,spat_dims)
    print(vec_field)


if __name__ == "__main__":
    main()