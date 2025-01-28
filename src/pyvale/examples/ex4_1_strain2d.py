'''
================================================================================
Example: strain gauges on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale

def main() -> None:
    """pyvale example: strain sensors on a 2D plate with a hole
    ----------------------------------------------------------------------------
    """
    data_path = pyvale.DataSet.mechanical_2d_output_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyvale.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    straingauge_array = pyvale.SensorArrayFactory \
                            .strain_gauges_basic_errs(sim_data,
                                                      sens_data,
                                                      "strain",
                                                      spat_dims=2)

    plot_field = 'strain_yy'
    pv_plot = pyvale.plot_point_sensors_on_sim(straingauge_array,plot_field)
    pv_plot.show(cpos="xy")

    pyvale.plot_time_traces(straingauge_array,'strain_xx')
    pyvale.plot_time_traces(straingauge_array,'strain_yy')
    pyvale.plot_time_traces(straingauge_array,'strain_xy')
    plt.show()


if __name__ == "__main__":
    main()