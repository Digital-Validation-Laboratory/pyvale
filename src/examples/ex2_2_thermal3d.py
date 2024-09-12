'''
================================================================================
Example: 3d thermocouples on a monoblock

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import mooseherder as mh
import pyvale


def main() -> None:
    """pyvale example:
    """
    # Use mooseherder to read the exodus and get a SimData object
    data_path = Path('src/simcases/case16/case16_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    field_name = 'temperature'

    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (1,4,1)
    x_lims = (11.5,11.5)
    y_lims = (0,31.0)
    z_lims = (0.0,12.5)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_array = pyvale.SensorArrayFactory() \
        .basic_thermocouple_array(sim_data,
                                  sens_pos,
                                  field_name,
                                  spat_dims=3)


    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')

    pv_plot = pyvale.plot_sensors_on_sim(tc_array,field_name)
    pv_plot.camera_position = [(52.198, 26.042, 60.099),
                                (0.0, 4.0, 5.5),
                                (-0.190, 0.960, -0.206)]
    pv_plot.show()


if __name__ == '__main__':
    main()
