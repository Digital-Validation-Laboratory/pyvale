'''
================================================================================
pycave: 2d thermocouples

authors: thescepticalrabbit
================================================================================
'''
from pprint import pprint
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import mooseherder as mh
import pycave


def main() -> None:
    data_path = Path('data/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    field_name = 'temperature'    # Same as in the moose input and SimData node_var key
    t_field = pycave.Field(sim_data,field_name,spat_dims)

    n_sens = (3,2,1)    # Number of sensor (x,y,z)
    x_lims = (0.0,2.0)  # Limits for each coord in sim length units
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pycave.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_freq = 5.0
    sample_times = np.arange(sim_data.time[0],sim_data.time[-1],1/sample_freq)

    tc_array = pycave.ThermocoupleArray(sens_pos,t_field,sample_times)

    tc_array.set_uniform_systematic_err_func(low=-10.0,high=10.0)
    tc_array.set_normal_random_err_func(std_dev=5.0)

    pv_sens = tc_array.get_visualiser()
    pv_sim = t_field.get_visualiser()
    pv_plot = pycave.plot_sensors(pv_sim,pv_sens,field_name)
    pv_plot.add_scalar_bar('Temperature, T [degC]')


    pv_plot_mode = 'off'

    if pv_plot_mode == 'interactive':
        pv_plot.show()
    if pv_plot_mode == 'save_fig':
        pv_plot.camera_position = [(-0.295, 1.235, 3.369),
                                (1.0274, 0.314, 0.0211),
                                (0.081, 0.969, -0.234)]
        save_render = Path('examples/images/plate_thermal_2d_sim_view.svg')
        pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex


    trace_plot_mode = 'interactive'

    (fig,ax) = tc_array.plot_time_traces(plot_truth=False,plot_sim=True)
    if trace_plot_mode == 'interactive':
        ax.set_xlim([0.0,5.0])
        plt.show()
    if trace_plot_mode == 'save_fig':
        save_traces = Path('examples/images/plate_thermal_2d_traces.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
