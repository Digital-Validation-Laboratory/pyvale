'''
================================================================================
Analytic test case data - linear

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import matplotlib.pyplot as plt
import numpy as np
import pyvale
import pyvale.visualisation.plotters

def main() -> None:
    (sim_data,data_gen) = pyvale.AnalyticCaseFactory.scalar_linear_2d()

    descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()

    field_key = 'scalar'
    t_field = pyvale.ScalarField(sim_data,
                                 field_key=field_key,
                                 spat_dim=2)

    n_sens = (4,2,1)
    x_lims = (0.0,10.0)
    y_lims = (0.0,7.5)
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

    errors_on = {'indep_sys': False,
                 'rand': False,
                 'dep_sys': False}

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

    pyvale.print_measurements(tc_array,
                            (0,1), # Sensor 1
                            (0,1), # Component 1: scalar field = 1 component
                            (measurements.shape[2]-5,measurements.shape[2]))

    (fig,_) = pyvale.plot_time_traces(tc_array,field_key)
    plt.show()

    pv_plot = pyvale.plot_sensors_on_sim(tc_array,field_key)
    pv_plot.show()

if __name__ == '__main__':
    main()
