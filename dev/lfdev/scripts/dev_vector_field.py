'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pprint import pprint
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

    dim = 2       # Specify that we only have 2 spatial dimensions
    field_name = 'disp'
    components = ('disp_x','disp_y')    # Same as in the moose input and SimData node_var key
    disp_field = pyvale.VectorField(sim_data,field_name,components,dim)


    n_sens = (2,3,1)    # Number of sensor (x,y,z)
    x_lims = (0,100e-3)  # Limits for each coord in sim length units
    y_lims = (0,150e-3)
    z_lims = (0.0,0.0)
    sample_points = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    samples = disp_field.sample_field(sample_points)
    print(sim_data.time)
    print(sim_data.time.shape)
    print(samples.shape)


if __name__ == "__main__":
    main()