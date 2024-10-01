'''
================================================================================
example: strain gauges on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale


def main() -> None:
    #===========================================================================
    # Load Simulations as mooseherder.SimData objects
    data_path = Path("src/data/case18_1_out.e")
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    #===========================================================================
    # Specify sensor locations and sample times
    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = True
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    #===========================================================================
    # Create pyvale sensor arrays for thermal and mechanical data
    tc_field = 'temperature'
    tc_array = pyvale.SensorArrayFactory \
        .basic_thermocouple_array(sim_data,
                                  sens_pos,
                                  tc_field,
                                  spat_dims=2,
                                  sample_times=None)

    sg_field = 'strain'
    sg_array = pyvale.SensorArrayFactory \
        .basic_straingauge_array(sim_data,
                                  sens_pos,
                                  sg_field,
                                  spat_dims=2,
                                  sample_times=None)

    #===========================================================================
    # Visualise Traces
    print(80*'-')
    sens_num = 4
    print('THERMAL: The last 5 time steps (measurements) of sensor {sens_num}:')
    pyvale.print_measurements(tc_array,
                              (sens_num-1,sens_num),
                              (0,1),
                              (tc_array.get_measurement_shape()[2]-5,
                               tc_array.get_measurement_shape()[2]))
    print(80*'-')

    pyvale.plot_time_traces(tc_array,"temperature")
    pyvale.plot_time_traces(sg_array,"strain_xx")
    plt.show()


if __name__ == "__main__":
    main()