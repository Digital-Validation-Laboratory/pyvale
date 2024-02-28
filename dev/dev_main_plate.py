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
    pprint(sens_pos.shape)

    t_field = pycave.Field(sim_data,field_name,dim)
    tc_array = pycave.ThermocoupleArray(sens_pos,t_field)

    err_range = 2.0
    rand_err_func = partial(np.random.default_rng().normal,
                            loc=0.0,
                            scale=err_range)
    sys_err_func = partial(np.random.default_rng().uniform,
                            low=-err_range,
                            high=err_range)

    tc_array.set_random_err_func(rand_err_func)
    tc_array.set_systematic_err_func(sys_err_func)

    sens_data = tc_array.get_measurement_data()

    sens_dict = asdict(sens_data)
    for ss in sens_dict:
        print()
        print(f'{ss} =')
        pprint(sens_dict[ss])

    pv_sens = tc_array.get_visualiser()
    pv_sim = t_field.get_visualiser()
    #pycave.plot_sensors(pv_sim,pv_sens,field_name)

    tc_array.plot_time_traces()

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

