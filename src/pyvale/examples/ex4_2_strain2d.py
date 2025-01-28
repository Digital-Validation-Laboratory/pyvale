'''
================================================================================
Example: strain gauges on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
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

    descriptor = pyvale.SensorDescriptor()
    descriptor.name = 'Strain'
    descriptor.symbol = r'\varepsilon'
    descriptor.units = r'-'
    descriptor.tag = 'SG'
    descriptor.components = ('xx','yy','xy')

    spat_dims = 2
    field_key = 'strain'
    norm_components = ('strain_xx','strain_yy')
    dev_components = ('strain_xy',)
    strain_field = pyvale.FieldTensor(sim_data,
                                    field_key,
                                    norm_components,
                                    dev_components,
                                    spat_dims)

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

    straingauge_array = pyvale.SensorArrayPoint(sens_data,
                                                strain_field,
                                                descriptor)

    error_chain = []
    error_chain.append(pyvale.ErrSysUniform(low=-0.1e-3,high=0.1e-3))
    error_chain.append(pyvale.ErrRandNormal(std=0.1e-3))
    error_int = pyvale.ErrIntegrator(error_chain,
                                       sens_data,
                                       straingauge_array.get_measurement_shape())
    straingauge_array.set_error_integrator(error_int)

    plot_field = 'strain_yy'
    pv_plot = pyvale.plot_point_sensors_on_sim(straingauge_array,plot_field)
    pv_plot.show(cpos="xy")

    pyvale.plot_time_traces(straingauge_array,'strain_xx')
    pyvale.plot_time_traces(straingauge_array,'strain_yy')
    pyvale.plot_time_traces(straingauge_array,'strain_xy')
    plt.show()


if __name__ == "__main__":
    main()