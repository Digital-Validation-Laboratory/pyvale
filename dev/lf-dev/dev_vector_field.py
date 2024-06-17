'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import pyvista as pv

class VectorField:
    pass


def main() -> None:
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    # Create a Field object that will allow the sensors to interpolate the sim
    # data field of interest quickly by using the mesh and shape functions
    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    field_name = 'temperature'    # Same as in the moose input and SimData node_var key
    t_field = pyvale.Field(sim_data,field_name,spat_dims)

    comps = ['','']
    #pv_points = pv.PolyData(sample_points)
    #sample_data = pv_points.sample(self._data_grid)
    #sample_data = np.array(sample_data[self._name]) # type: ignore



if __name__ == "__main__":
    main()