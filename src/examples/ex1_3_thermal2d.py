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
    - Explanation of the different types of error models
    - There are flags throughout the example allowing the user to toggle on/off
      parts of the sensor array construction

    NOTES:
    - Systematic  errors are assumed to be constant for all time steps for point
      sensors
    - Random errors are sampled for all sensors (i.e. positions) and times when
      the `calc` methods are called
    - Independent systematic errors are calculated from the truth value if
      required
    - Random errors calculated based on the truth value if required
    - Dependent systematic errors calculated based on the current integrated
      measurement value at that step
    -
    """
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()


    use_auto_descriptor = 'blank'
    if use_auto_descriptor == 'factory':
        descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()
    elif use_auto_descriptor == 'manual':
        descriptor = pyvale.SensorDescriptor()
        descriptor.name = 'Temperature'
        descriptor.symbol = 'T'
        descriptor.units = r'^{\circ}C'
        descriptor.tag = 'TC'
    else:
        descriptor = pyvale.SensorDescriptor()

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
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)


    tc_array = pyvale.PointSensorArray(sens_pos,
                                       t_field,
                                       sample_times,
                                       descriptor)

    errors_on = {'indep_sys': True,
                 'rand': False,
                 'dep_sys': True}

    if errors_on['indep_sys']:
        indep_sys_err1 = pyvale.SysErrOffset(offset=-5.0)
        indep_sys_err2 = pyvale.SysErrUniform(low=-10.0,
                                            high=10.0)
        indep_sys_err3 = pyvale.SysErrPointPosition(t_field,
                                            sens_pos,
                                            (0.05,0.05,None),
                                            sample_times)
        indep_sys_err_int = pyvale.ErrorIntegrator([indep_sys_err1,
                                                    indep_sys_err2,
                                                    indep_sys_err3],
                                            tc_array.get_measurement_shape())

        tc_array.set_indep_sys_err_integrator(indep_sys_err_int)

    if errors_on['rand']:
        rand_err1 = pyvale.RandErrNormPercent(std_percent=5.0)
        rand_err2 = pyvale.RandErrUnifPercent(low_percent=-5.0,
                                            high_percent=5.0)
        rand_err_int = pyvale.ErrorIntegrator([rand_err1,rand_err2],
                                                tc_array.get_measurement_shape())
        tc_array.set_rand_err_integrator(rand_err_int)

    if errors_on['dep_sys']:
        dep_sys_err1 = pyvale.SysErrDigitisation(bits_per_unit=1/10)
        dep_sys_err2 = pyvale.SysErrSaturation(meas_min=0.0,meas_max=300.0)
        dep_sys_err_int = pyvale.ErrorIntegrator([dep_sys_err1,dep_sys_err2],
                                            tc_array.get_measurement_shape())
        tc_array.set_dep_sys_err_integrator(dep_sys_err_int)

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

    pyvale.plot_time_traces(tc_array,field_key)
    plt.show()


if __name__ == '__main__':
    main()
