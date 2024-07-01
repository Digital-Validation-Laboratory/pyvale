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
    # Use mooseherder to read the exodus and get a SimData object
    data_path = Path('simcases/case17/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    # Create a Field object that will allow the sensors to interpolate the sim
    # data field of interest quickly by using the mesh and shape functions
    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    field_name = 'displacement'
    components = ('disp_x','disp_y')
    disp_field = pyvale.VectorField(sim_data,field_name,components,spat_dims)

    # This creates a grid of sensors
    n_sens = (2,3,1)    # Number of sensor (x,y,z)
    x_lims = (0.0,100.0)  # Limits for each coord in scaled sim length units
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    disp_sens_array = pyvale.PointSensorArray(sens_pos,disp_field)

    err_sys1 = pyvale.SysErrUniform(low=-0.01e-3,high=0.01e-3)
    sys_err_int = pyvale.ErrorIntegrator([err_sys1],
                                          disp_sens_array.get_measurement_shape())
    disp_sens_array.set_pre_sys_err_integrator(sys_err_int)

    err_rand1 = pyvale.RandErrNormal(std=0.01e-3)
    rand_err_int = pyvale.ErrorIntegrator([err_rand1],
                                            disp_sens_array.get_measurement_shape())
    disp_sens_array.set_rand_err_integrator(rand_err_int)

    plot_field = 'off'
    if plot_field == 'disp_x':
        pv_plot = pyvale.plot_sensors_on_sim(disp_sens_array,'disp_x')
        pv_plot.add_scalar_bar(r'Displacement X [m]')
        pv_plot.show()
    elif plot_field == 'disp_y':
        pv_plot = pyvale.plot_sensors_on_sim(disp_sens_array,'disp_y')
        pv_plot.add_scalar_bar(r'Displacement Y [m]')
        pv_plot.show()

    (fig,_) = pyvale.plot_time_traces(disp_sens_array,'disp_x')
    (fig,_) = pyvale.plot_time_traces(disp_sens_array,'disp_y')
    plt.show()


if __name__ == "__main__":
    main()