'''
================================================================================
example: displacement sensors on a 2d plate

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
    """pyvale example:
    """
    data_path = Path('simcases/case17/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    disp_sens_array = pyvale.SensorArrayFactory \
                            .basic_dispsens_array(sim_data,
                                                sens_pos,
                                                "displacement",
                                                spat_dims=2)

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