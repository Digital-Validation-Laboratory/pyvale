'''
================================================================================
example: thermocouples on a 2d plate

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
    """Pyvale example: Point sensors on a 2D thermal simulation
    ----------------------------------------------------------------------------
    - Quick start
    - Basic sensor array construction using the sensor array factory
    - Basic visualisation of sensor locations and sensor traces with the pyvale
      wrapper for pyvista and matplotlib.
    """

    data_path = Path('src/data/case13_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_key = list(sim_data.node_vars.keys())[0] # type: ignore

    n_sens = (3,2,1)
    x_lims = (0.0,2.0)
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_array = pyvale.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_pos,
                                  field_key,
                                  spat_dims=2)

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for last sensor:\n{measurements[-1,0,:]}\n')

    pv_plot = pyvale.plot_point_sensors_on_sim(tc_array,field_key)
    # Set this to 'interactive' to get an interactive 3D plot of the simulation
    # and labelled sensor locations, set to 'save_fig' to create a vector
    # graphic using a specified camera position.
    pv_plot_mode = 'interactive'

    if pv_plot_mode == 'interactive':
        pv_plot.camera_position = [(-0.295, 1.235, 3.369),
                                    (1.0274, 0.314, 0.0211),
                                    (0.081, 0.969, -0.234)]
        pv_plot.show(cpos="xy")

        print(80*"=")
        print('Camera positions = ')
        print(pv_plot.camera_position)
        print(80*"="+"\n")

    if pv_plot_mode == 'save_fig':
        # Determined manually by moving camera and then dumping camera position
        # to console after window close - see 'interactive above'
        pv_plot.camera_position = [(-0.295, 1.235, 3.369),
                                (1.0274, 0.314, 0.0211),
                                (0.081, 0.969, -0.234)]
        save_render = Path('src/examples/plate_thermal_2d_sim_view.svg')
        pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
        pv_plot.screenshot(save_render.with_suffix('.png'))

    # Set this to 'interactive' to get a matplotlib.pyplot with the sensor
    # traces plotted over time. Set to 'save_fig' to save an image of the plot
    # to file.
    trace_plot_mode = 'interactive'

    (fig,_) = pyvale.plot_time_traces(tc_array,field_key)

    if trace_plot_mode == 'interactive':
        plt.show()
    if trace_plot_mode == 'save_fig':
        save_traces = Path('src/examples/plate_thermal_2d_traces.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
