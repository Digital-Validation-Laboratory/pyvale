'''
================================================================================
Example: 3d thermocouples on a monoblock

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from pathlib import Path
import mooseherder as mh
import pyvale


def main() -> None:
    """pyvale example:
    """
    # Use mooseherder to read the exodus and get a SimData object
    data_path = Path('src/data/case16_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    field_name = 'temperature'

    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0,31.0)
    z_lims = (0.0,12.5)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = pyvale.SensorData(positions=sens_pos)

    tc_array = pyvale.SensorArrayFactory() \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  field_name,
                                  spat_dims=3)


    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')

    pv_plot = pyvale.plot_sensors_on_sim(tc_array,field_name)
    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                                (-2.858, 13.189, 4.523),
                                (-0.215, 0.948, -0.233)]
    pv_plot.show()


if __name__ == '__main__':
    main()
