'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale


def main() -> None:
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    field_name = 'temperature'    # Same as in the moose input and SimData node_var key
    t_field = pyvale.ScalarField(sim_data,field_name,spat_dims)

    n_sens = (3,2,1)    # Number of sensor (x,y,z)
    x_lims = (0.0,2.0)  # Limits for each coord in sim length units
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_array = pyvale.ThermocoupleArray(sens_pos,t_field)

    err_unif1 = pyvale.SysErrUniform(low=10.0,high=20.0)
    err_unif2 = pyvale.SysErrUniform(low=-5.0,high=5.0)
    sys_err_int = pyvale.SysErrIntegrator([err_unif1,err_unif2],
                                          tc_array.get_measurement_shape())
    tc_array.set_sys_err_integrator(sys_err_int)


    #tc_array.set_normal_random_err_func(std_dev=1.0)
    #measurements = tc_array.get_measurements()

    pv_sens = tc_array.get_visualiser()
    pv_sim = t_field.get_visualiser()
    pv_plot = pyvale.plot_sensors(pv_sim,pv_sens,field_name)
    pv_plot.add_scalar_bar('Temperature, T [degC]')


    pv_plot_mode = 'off'

    if pv_plot_mode == 'interactive':
        pv_plot.show()
        pprint('Camera positions = ')
        pprint(pv_plot.camera_position)
    if pv_plot_mode == 'save_fig':
        pv_plot.camera_position = [(-0.295, 1.235, 3.369),
                                (1.0274, 0.314, 0.0211),
                                (0.081, 0.969, -0.234)]
        save_render = Path('examples/images/plate_thermal_2d_sim_view.svg')
        pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
        pv_plot.screenshot(save_render.with_suffix('.png'))


    trace_plot_mode = 'interactive'

    (fig,_) = tc_array.plot_time_traces(plot_truth=True)
    if trace_plot_mode == 'interactive':
        plt.show()
    if trace_plot_mode == 'save_fig':
        save_traces = Path('examples/images/plate_thermal_2d_traces.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
