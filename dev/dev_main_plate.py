'''
================================================================================
pycave: dev_main

authors: thescepticalrabbit
================================================================================
'''
from pprint import pprint
from pathlib import Path
from functools import partial
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

import mooseherder as mh
from pyvista.plotting.opts import ElementType

import pycave
from pycave.plotprops import PlotProps


def main() -> None:
    data_path = Path('data/thermal_2d_basic_out.e')
    dim = 2
    field_name = 'T'

    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    x_sens = 4
    x_min = 0
    x_max = 2

    y_sens = 1
    y_min = 0
    y_max = 1

    z_sens = 1
    z_min = 0
    z_max = 0

    sens_pos_x = np.linspace(x_min,x_max,x_sens+2)[1:-1]
    sens_pos_y = np.linspace(y_min,y_max,y_sens+2)[1:-1]
    sens_pos_z = np.linspace(z_min,z_max,z_sens+2)[1:-1]
    (sens_grid_x,sens_grid_y,sens_grid_z) = np.meshgrid(
        sens_pos_x,sens_pos_y,sens_pos_z)

    sens_pos_x = sens_grid_x.flatten()
    sens_pos_y = sens_grid_y.flatten()
    sens_pos_z = sens_grid_z.flatten()
    sens_pos = np.vstack((sens_pos_x,sens_pos_y,sens_pos_z)).T

    t_field = pycave.Field(sim_data,field_name,dim)
    tc_array = pycave.ThermocoupleArray(sens_pos,t_field)

    err_range = 2.0
    rand_err_func = partial(np.random.default_rng().normal,
                            loc=0.0,
                            scale=err_range)

    def sys_err_func(size: tuple) -> np.ndarray:
        # Assume the first index of the tuple is the number of sensors
        err_range = 20.0
        sys_errs = np.random.default_rng().uniform(low=-err_range,
                                                   high=err_range,
                                                   size=(size[0],1))
        #print(sys_errs.shape)
        sys_errs = np.tile(sys_errs,(1,size[1]))
        #print(sys_errs.shape)
        return sys_errs


    tc_array.set_random_err_func(rand_err_func)
    tc_array.set_systematic_err_func(sys_err_func)

    sens_data = tc_array.get_measurement_data()

    #sens_dict = asdict(sens_data)

    pv_sens = tc_array.get_visualiser()
    pv_sim = t_field.get_visualiser()
    pv_plot = pycave.plot_sensors(pv_sim,pv_sens,field_name)

    pv_plot.add_scalar_bar('Temperature, T [degC]')
    #pv_plot.add_text(r'$\rho$', position='upper_left', font_size=150, color='blue')

    set_iso_view = False
    if set_iso_view:
        pv_plot.view_isometric()
    else:
        # Determined manually by moving camera and then dumping camera position to
        # console after window close
        pv_plot.camera_position = [(-0.295, 1.235, 3.369),
                                (1.0274, 0.314, 0.0211),
                                (0.081, 0.969, -0.234)]
    save_render = Path('dev/images/plate_thermal_2d_sim_view.pdf')
    pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
    #pprint(pv_plot.camera_position)
    pv_plot.close()

    (fig,ax) = tc_array.plot_time_traces(plot_truth=True)
    plt.show()
    save_traces = save_render.with_name('plate_thermal_2d_traces').with_suffix('.png')
    fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

