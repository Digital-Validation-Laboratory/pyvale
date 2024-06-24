'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Callable, Any
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from pyvale.field import Field
from pyvale.sensorarray import SensorArray, MeasurementData
from pyvale.plotprops import PlotProps


class DispSensArray(SensorArray):
    def __init__(self,
                 positions: np.ndarray,
                 field: Field,
                 sample_times: np.ndarray | None = None
                 ) -> None:

        self._positions = positions
        self._field = field
        self._sample_times = sample_times

        self._sys_err_int = None
        self._rand_err_int = None


    def get_positions(self) -> np.ndarray:
        return self._positions

    def get_sample_times(self) -> np.ndarray:
        if self._sample_times is None:
            return self._field.get_time_steps()

        return self._sample_times

    def get_num_sensors(self) -> int:
        return self._positions.shape[0]

    def get_measurement_shape(self) -> tuple[int,int,int]:
        return (self.get_num_sensors(),
                len(self._field.get_all_components()),
                self.get_sample_times().shape[0])

    #---------------------------------------------------------------------------
    # Truth values - from simulation
    def get_truth_values(self) -> np.ndarray:
        return self._field.sample_field(self._positions,
                                        self._sample_times)


    def get_systematic_errs(self) -> np.ndarray:
        pass


    def get_random_errs(self) -> np.ndarray:
        pass


    def get_measurements(self) -> np.ndarray:
        pass


    def get_measurement_data(self) -> MeasurementData:
        pass

