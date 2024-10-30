'''
================================================================================
Example: strain gauges on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from pathlib import Path
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale

def main() -> None:
    data_path = Path('src/data/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    straingauge_array = pyvale.SensorArrayFactory \
                            .basic_straingauge_array(sim_data,
                                                     sens_pos,
                                                     "strain",
                                                     spat_dims=2)

    plot_field = 'strain_yy'
    pv_plot = pyvale.plot_sensors_on_sim(straingauge_array,plot_field)
    pv_plot.show()

    pyvale.plot_time_traces(straingauge_array,'strain_xx')
    pyvale.plot_time_traces(straingauge_array,'strain_yy')
    pyvale.plot_time_traces(straingauge_array,'strain_xy')
    plt.show()


if __name__ == "__main__":
    main()