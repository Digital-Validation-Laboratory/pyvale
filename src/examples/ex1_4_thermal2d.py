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
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale


def main() -> None:
    """Pyvale example: Point sensors on a 2D thermal simulation
    ----------------------------------------------------------------------------
    -
    """
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    field_key = list(sim_data.node_vars.keys())[0] # type: ignore

    n_sens = (4,1,1)
    x_lims = (0.0,2.0)
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),12)

    tc_array = pyvale.SensorArrayFactory \
        .basic_thermocouple_array(sim_data,
                                  sens_pos,
                                  field_key,
                                  spat_dims=2,
                                  sample_times=sample_times)

    indep_sys_err_int = pyvale.ErrorIntegrator([pyvale.SysErrOffset(offset=-20.0)],
                                        tc_array.get_measurement_shape())
    tc_array.set_indep_sys_err_integrator(indep_sys_err_int)

    measurements = tc_array.get_measurements()

    print(80*'-')
    print('Looking at the last 5 time steps (measurements) of sensor 0:')
    pyvale.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')




    trace_props = pyvale.SensorTraceOpts()

    trace_props.truth_line = None
    trace_props.sim_line = None
    pyvale.plot_time_traces(tc_array,field_key,trace_props)

    trace_props.meas_line = '--o'
    trace_props.truth_line = '-x'
    trace_props.sim_line = ':+'
    pyvale.plot_time_traces(tc_array,field_key,trace_props)

    trace_props.sensors_to_plot = np.arange(measurements.shape[0]-2
                                           ,measurements.shape[0])
    pyvale.plot_time_traces(tc_array,field_key,trace_props)

    trace_props.sensors_to_plot = None
    trace_props.time_min_max = (5.0,25.0)
    pyvale.plot_time_traces(tc_array,field_key,trace_props)

    plt.show()

    pv_plot = pyvale.plot_sensors_on_sim(tc_array,field_key)
    pv_plot.camera_position = [(-0.295, 1.235, 3.369),
                                (1.0274, 0.314, 0.0211),
                                (0.081, 0.969, -0.234)]
    pv_plot.show()


if __name__ == '__main__':
    main()
