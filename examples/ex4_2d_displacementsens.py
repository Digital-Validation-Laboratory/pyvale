'''
================================================================================
example: displacement sensors on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import mooseherder as mh
import pyvale

def main() -> None:
    # Use mooseherder to read the exodus and get a SimData object
    data_path = Path('simcases/case17/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    # Create a Field object that will allow the sensors to interpolate the sim
    # data field of interest quickly by using the mesh and shape functions
    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    field_name = 'displacement'
    components = ('disp_x','disp_y')
    disp_field = pyvale.VectorField(sim_data,field_name,components,spat_dims)





if __name__ == "__main__":
    main()