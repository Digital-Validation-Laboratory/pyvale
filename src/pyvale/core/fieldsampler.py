"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
import pyvista as pv
from pyvale.core.field import IField
from pyvale.core.sensordata import SensorData
from pyvale.core.integratorfactory import build_spatial_averager


def sample_field_with_sensor_data(field: IField, sensor_data: SensorData
                                  ) -> np.ndarray:
    """Samples (interpolated) and `IField` object using the parameters specified
    in the `SensorData` object.

    Parameters
    ----------
    field : IField
        Interface for
    sensor_data : SensorData
        Contains sensor array parameters including: number of sensors, positions
        and sample times. See the `SensorData` class for more information.

    Returns
    -------
    np.ndarray
        Array
    """
    if sensor_data.spatial_averager is None:
        return field.sample_field(sensor_data.positions,
                                  sensor_data.sample_times,
                                  sensor_data.angles)

    spatial_integrator = build_spatial_averager(field,sensor_data)
    return spatial_integrator.calc_averages()


# NOTE: sampling outside the bounds of the sample returns a value of 0
def sample_pyvista_grid(components: tuple,
                pyvista_grid: pv.UnstructuredGrid,
                time_steps: np.ndarray,
                points: np.ndarray,
                times: np.ndarray | None = None
                ) -> np.ndarray:

    # Use pyvista and shape functions for spatial interpolation at sim times
    pv_points = pv.PolyData(points)
    sample_data = pv_points.sample(pyvista_grid)

    # Push into the measurement array, shape=(n_sensors,n_comps,n_time_steps)
    n_comps = len(components)
    (n_sensors,n_time_steps) = np.array(sample_data[components[0]]).shape
    sample_at_sim_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_sim_time[:,ii,:] = np.array(sample_data[cc])

    # If sensor times are sim times then we return
    if times is None:
        return sample_at_sim_time

    # Use linear interpolation to extract sensor times
    def sample_time_interp(x):
        return np.interp(times, time_steps, x)

    n_time_steps = times.shape[0]
    sample_at_spec_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_spec_time[:,ii,:] = np.apply_along_axis(sample_time_interp,-1,
                                                    sample_at_sim_time[:,ii,:])

    return sample_at_spec_time



