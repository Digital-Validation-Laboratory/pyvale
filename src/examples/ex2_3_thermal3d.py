'''
================================================================================
Example: 3d thermocouples on a monoblock

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
    """pyvale example:
    """
    data_path = Path('src/data/case16_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    use_auto_descriptor = 'manual'
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
                                 spat_dim=3)

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0.0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),80)

    sens_data = pyvale.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    tc_array = pyvale.PointSensorArray(sens_data,
                                       t_field,
                                       descriptor)

    errors_on = {'indep_sys': True,
                 'rand': True,
                 'dep_sys': True}

    error_chain = []
    if errors_on['indep_sys']:
        error_chain.append(pyvale.SysErrOffset(offset=-5.0))
        error_chain.append(pyvale.SysErrUniform(low=-10.0,
                                                   high=10.0))

    if errors_on['rand']:
        error_chain.append(pyvale.RandErrNormPercent(std_percent=5.0))
        error_chain.append(pyvale.RandErrUnifPercent(low_percent=-5.0,
                                                   high_percent=5.0))

    if errors_on['dep_sys']:
        error_chain.append(pyvale.SysErrDigitisation(bits_per_unit=1/20))
        error_chain.append(pyvale.SysErrSaturation(meas_min=0.0,meas_max=800.0))

    if len(error_chain) > 0:
        error_integrator = pyvale.ErrorIntegrator(error_chain,
                                                  tc_array.get_measurement_shape())
        tc_array.set_error_integrator(error_integrator)

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')


    (fig,_) = pyvale.plot_time_traces(tc_array,field_key)
    plt.show()

    save_fig = True
    if save_fig:
        save_traces = Path('src/examples/monoblock_thermal_traces_syserrs.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')



if __name__ == '__main__':
    main()
