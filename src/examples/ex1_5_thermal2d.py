'''
================================================================================
example: thermocouples on a 2d plate

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
    """pyvale example: Point sensors on a 2D thermal simulation
    ----------------------------------------------------------------------------
    """
    data_path = Path('src/data/case13_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    field_key = list(sim_data.node_vars.keys())[0] # type: ignore
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    tc_array = pyvale.SensorArrayFactory \
        .plain_thermocouple_array(sim_data,
                                  sens_pos,
                                  field_key,
                                  spat_dims=2,
                                  sample_times=sample_times)

    #===========================================================================
    # Examples of full error library

    #---------------------------------------------------------------------------
    # Standard independent systematic errors
    pre_sys_errs = []
    pre_sys_errs.append(pyvale.SysErrOffset(offset=-1.0))
    pre_sys_errs.append(pyvale.SysErrOffsetPercent(offset_percent=-1.0))

    pre_sys_errs.append(pyvale.SysErrUniform(low=-2.0,
                                        high=2.0))
    pre_sys_errs.append(pyvale.SysErrUniformPercent(low_percent=-2.0,
                                                high_percent=2.0))

    pre_sys_errs.append(pyvale.SysErrNormal(std=1.0))
    pre_sys_errs.append(pyvale.SysErrNormPercent(std_percent=2.0))

    sys_gen = pyvale.TriangularGenerator(left=-1.0,
                                          mode=0.0,
                                          right=1.0)
    pre_sys_errs.append(pyvale.SysErrGenerator(sys_gen))

    # Field based errors
    pos_gen = pyvale.NormalGenerator(std=1.0)
    pre_sys_errs.append(pyvale.SysErrRandPosition(tc_array.field,
                                                  sens_pos,
                                                  (pos_gen,pos_gen,None),
                                                  sample_times))


    indep_sys_err_int = pyvale.ErrorIntegrator(pre_sys_errs,
                        tc_array.get_measurement_shape())
    tc_array.set_indep_sys_err_integrator(indep_sys_err_int)

    #---------------------------------------------------------------------------
    rand_errs = []
    rand_errs.append(pyvale.RandErrNormal(std = 2.0))
    rand_errs.append(pyvale.RandErrNormPercent(std_percent=2.0))

    rand_errs.append(pyvale.RandErrUniform(low=-2.0,high=2.0))
    rand_errs.append(pyvale.RandErrUnifPercent(low_percent=-2.0,
                                               high_percent=2.0))

    rand_gen = pyvale.TriangularGenerator(left=-5.0,
                                          mode=0.0,
                                          right=5.0)
    rand_errs.append(pyvale.RandErrGenerator(rand_gen))

    rand_err_int = pyvale.ErrorIntegrator(rand_errs,
                                          tc_array.get_measurement_shape())
    tc_array.set_rand_err_integrator(rand_err_int)

    #---------------------------------------------------------------------------
    post_sys_errs = []
    post_sys_errs.append(pyvale.SysErrDigitisation(bits_per_unit=2**8/100))
    post_sys_errs.append(pyvale.SysErrSaturation(meas_min=0.0,meas_max=300.0))

    dep_sys_err_int = pyvale.ErrorIntegrator(post_sys_errs,
                                             tc_array.get_measurement_shape())
    tc_array.set_dep_sys_err_integrator(dep_sys_err_int)


    #===========================================================================

    measurements = tc_array.calc_measurements()
    print(80*'-')
    sens_num = 4
    print('The last 5 time steps (measurements) of sensor {sens_num}:')
    pyvale.print_measurements(tc_array,
                              (sens_num-1,sens_num),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')

    pyvale.plot_time_traces(tc_array,field_key)
    plt.show()


if __name__ == '__main__':
    main()