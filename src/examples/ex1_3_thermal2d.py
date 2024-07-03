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
    - Full construction of a point sensor array from scratch
    - Explanation of the different

    """
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    use_auto_descriptor = True
    if use_auto_descriptor:
        descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()
    else:
        descriptor = pyvale.SensorDescriptor()
        descriptor.name = 'Temperature'
        descriptor.symbol = 'T'
        descriptor.units = r'^{\circ}C'
        descriptor.tag = 'TC'

    field_key = 'temperature'
    t_field = pyvale.ScalarField(sim_data,
                                 field_key=field_key,
                                 spat_dim=2)

    n_sens = (4,1,1)
    x_lims = (0.0,2.0)
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),10)


    tc_array = pyvale.PointSensorArray(sens_pos,
                                       t_field,
                                       sample_times,
                                       descriptor)


    pre_sys_err1 = pyvale.SysErrOffset(offset=-10.0)
    pre_sys_err1 = pyvale.SysErrOffset(offset=-10.0)
    pre_sys_err_int = pyvale.ErrorIntegrator([pre_sys_err1],
                                          tc_array.get_measurement_shape())
    tc_array.set_pre_sys_err_integrator(pre_sys_err_int)
    '''
    rand_err1 = pyvale.RandErrNormal(std=10.0)
    rand_err2 = pyvale.RandErrUniform(low=-10.0,high=10.0)
    rand_err_int = pyvale.ErrorIntegrator([rand_err1,rand_err2],
                                            tc_array.get_measurement_shape())
    tc_array.set_rand_err_integrator(rand_err_int)

    post_sys_err1 = pyvale.SysErrDigitisation(bits_per_unit=1/10)
    post_sys_err2 = pyvale.SysErrSaturation(meas_min=0.0,meas_max=350.0)
    post_sys_err_int = pyvale.ErrorIntegrator([post_sys_err1,post_sys_err2],
                                        tc_array.get_measurement_shape())
    tc_array.set_post_sys_err_integrator(post_sys_err_int)
    '''

    measurements = tc_array.get_measurements()

    print('\n'+80*'-')
    print('For a sensor: measurement = truth + sysematic error + random error')
    print(f'measurements.shape = {measurements.shape} = '+
          '(n_sensors,n_field_components,n_timesteps)\n')
    print("The truth, systematic error and random error arrays have the same "+
          "shape.")

    print(80*'-')
    print('Looking at the last 5 time steps (measurements) of sensor 0:')
    pyvale.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')

    plot_on = True
    if plot_on:
        trace_props = pyvale.SensorPlotOpts()
        trace_props.truth_line = '-^'
        trace_props.sim_line = '-x'
        pyvale.plot_time_traces(tc_array,field_key,trace_props)
        plt.show()


if __name__ == '__main__':
    main()
