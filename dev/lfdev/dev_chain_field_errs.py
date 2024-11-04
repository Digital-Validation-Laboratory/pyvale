'''
================================================================================
example: displacement sensors on a 2d plate

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
    data_path = Path('src/data/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptor()
    descriptor.name = 'Displacement'
    descriptor.symbol = r'u'
    descriptor.units = r'm'
    descriptor.tag = 'DS'
    descriptor.components = ('x','y','z')

    spat_dims = 2
    field_key = 'disp'
    components = ('disp_x','disp_y')
    disp_field = pyvale.VectorField(sim_data,field_key,components,spat_dims)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sensor_positions = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = True
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = pyvale.SensorData(positions=sensor_positions,
                                    sample_times=sample_times)

    disp_sens_array = pyvale.PointSensorArray(sensor_data,
                                              disp_field,
                                              descriptor)

    pos_offset = -1.0*np.ones_like(sensor_positions)
    pos_offset[:,2] = 0 # in 2d we only have offset in x and y so zero z
    pos_error_data = pyvale.FieldErrorData(pos_offset_xyz=pos_offset)
    angle_offset = np.zeros_like(sensor_positions)
    angle_offset[:,0] = 5.0 # only rotate about z in 2D
    time_offset = 1.0*np.ones_like(disp_sens_array.get_sample_times())
    time_error_data = pyvale.FieldErrorData(time_offset=time_offset)

    field_errs = []

    field_errs.append(pyvale.SysErrField(disp_field,
                                        time_error_data))
    field_errs.append(pyvale.SysErrField(disp_field,
                                        time_error_data))

    err_int_opts = pyvale.ErrorIntegrationOpts(force_dependence=True,
                                               store_errs_by_func=True)
    error_int = pyvale.ErrorIntegrator(field_errs,
                                       sensor_data,
                                       disp_sens_array.get_measurement_shape(),
                                       err_int_opts)
    disp_sens_array.set_error_integrator(error_int)

    measurements = disp_sens_array.calc_measurements()

    print(sensor_data.sample_times)
    print()

    sens_data_by_chain = error_int.get_sens_data_by_chain()
    if sens_data_by_chain is not None:
        for ii,ss in enumerate(sens_data_by_chain):
            if ss is not None:
                print(f"SensorData @ [{ii}]")
                print(ss.sample_times)

    sens_data_accumulated = error_int.get_sens_data_accumulated()
    print()
    print(sens_data_accumulated.sample_times)
    print()

    return

    print(80*'-')
    sens_num = 4
    print('The last 5 time steps (measurements) of sensor {sens_num}:')
    pyvale.print_measurements(disp_sens_array,
                              (sens_num-1,sens_num),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')

    sens_data_by_chain = error_int.get_sens_data_by_chain()
    for ii,ss in enumerate(sens_data_by_chain):
        if ss is not None:
            print(f"SensorData @ [{ii}]")
            print(ss)

    plot_field = 'disp_x'
    if plot_field == 'disp_x':
        pv_plot = pyvale.plot_sensors_on_sim(disp_sens_array,'disp_x')
        pv_plot.show()
    elif plot_field == 'disp_y':
        pv_plot = pyvale.plot_sensors_on_sim(disp_sens_array,'disp_y')
        pv_plot.show()

    pyvale.plot_time_traces(disp_sens_array,'disp_x')
    pyvale.plot_time_traces(disp_sens_array,'disp_y')
    plt.show()


if __name__ == "__main__":
    main()