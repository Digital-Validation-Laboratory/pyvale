'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale


def main() -> None:
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    spat_dims = 2
    field_name = 'temperature'
    t_field = pyvale.ScalarField(sim_data,field_name,spat_dims)

    n_sens = (4,1,1)
    x_lims = (0.0,2.0)
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_array = pyvale.PointSensorArray(sens_pos,t_field)

    pre_syserrs_on = False
    if pre_syserrs_on:
        err_sys1 = pyvale.SysErrUniform(low=-20.0,high=20.0)
        err_sys2 = pyvale.SysErrNormal(std=20.0)
        pre_syserr_int = pyvale.ErrorIntegrator([err_sys1,err_sys2],
                                            tc_array.get_measurement_shape())
        tc_array.set_pre_sys_err_integrator(pre_syserr_int)

    randerrs_on = False
    if randerrs_on:
        err_rand1 = pyvale.RandErrNormal(std=10.0)
        err_rand2 = pyvale.RandErrUniform(low=-10.0,high=10.0)
        rand_err_int = pyvale.ErrorIntegrator([err_rand1,err_rand2],
                                                tc_array.get_measurement_shape())
        tc_array.set_rand_err_integrator(rand_err_int)


    #post_syserr1 = pyvale.SysErrRoundOff(method='round',base=5)
    post_syserr_ints = [#pyvale.SysErrOffset(offset=5.0),
                        pyvale.SysErrOffset(offset=5.0),
                        pyvale.SysErrDigitisation(bits_per_unit=(2**8/2560)),
                        pyvale.SysErrSaturation(meas_min=20.0,meas_max=300.0)]
    post_syserr_int = pyvale.ErrorIntegrator(post_syserr_ints,
                                             tc_array.get_measurement_shape())
    tc_array.set_post_sys_err_integrator(post_syserr_int)


    measurements = tc_array.calc_measurements()

    print_meas = True
    if print_meas:
        print(80*'-')
        pyvale.print_measurements(tc_array,
                                (measurements.shape[0]-1,measurements.shape[0]),
                                (0,1),
                                (measurements.shape[2]-10,measurements.shape[2]))
        print(80*'-')

        pyvale.plot_time_traces(tc_array,field_name)
        plt.show()


if __name__ == '__main__':
    main()
