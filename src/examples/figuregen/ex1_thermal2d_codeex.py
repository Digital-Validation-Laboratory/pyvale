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

data_path = Path('data/examplesims/plate_2d_thermal_out.e')
data_reader = mh.ExodusReader(data_path)
sim_data = data_reader.read_all_sim_data()

FIELD_KEY = 'temperature'
temperature_field = pyvale.ScalarField(sim_data,
                             field_key=FIELD_KEY,
                             spat_dim=2)

n_sens = (4,1,1)
x_lims = (0.0,2.0)
y_lims = (0.0,1.0)
z_lims = (0.0,0.0)
sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

sample_times = np.linspace(0.0,np.max(sim_data.time),50)

descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()

thermocouple_array = pyvale.PointSensorArray(sens_pos,
                                            temperature_field,
                                            sample_times,
                                            descriptor)


indep_sys_err1 = pyvale.SysErrOffset(offset=-5.0)
indep_sys_err2 = pyvale.SysErrUniform(low=-10.0,
                                    high=10.0)
indep_sys_err_int = pyvale.ErrorIntegrator([indep_sys_err1,indep_sys_err2],
                                    thermocouple_array.get_measurement_shape())
thermocouple_array.set_indep_sys_err_integrator(indep_sys_err_int)


rand_err1 = pyvale.RandErrNormPercent(std_percent=5.0)
rand_err2 = pyvale.RandErrUnifPercent(low_percent=-5.0,
                                    high_percent=5.0)
rand_err_int = pyvale.ErrorIntegrator([rand_err1,rand_err2],
                                        thermocouple_array.get_measurement_shape())
thermocouple_array.set_rand_err_integrator(rand_err_int)


dep_sys_err1 = pyvale.SysErrDigitisation(bits_per_unit=1/10)
dep_sys_err2 = pyvale.SysErrSaturation(meas_min=0.0,meas_max=300.0)
dep_sys_err_int = pyvale.ErrorIntegrator([dep_sys_err1,dep_sys_err2],
                                    thermocouple_array.get_measurement_shape())
thermocouple_array.set_dep_sys_err_integrator(dep_sys_err_int)



measurements = thermocouple_array.get_measurements()

NUM_EXPERIMENTS = 3
for ee in range(NUM_EXPERIMENTS):
    thermocouple_array.calc_measurements()

    pyvale.print_measurements(thermocouple_array,
                            (0,1),
                            (0,1),
                            (measurements.shape[2]-5,measurements.shape[2]))

    pyvale.plot_time_traces(thermocouple_array,FIELD_KEY)

plt.show()







