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

        self._sys_err_func = None
        self._sys_errs = None

        self._rand_err_func = None

    def get_positions(self) -> np.ndarray:
        return self._positions

    def get_sample_times(self) -> np.ndarray:
        if self._sample_times is None:
            return self._field.get_time_steps()

        return self._sample_times

    #---------------------------------------------------------------------------
    # Truth values - from simulation
    def get_truth_values(self) -> dict[str,np.ndarray]:
        return self._field.sample_field(self._positions,
                                        self._sample_times)


    def get_systematic_errs(self) -> dict[str,np.ndarray]:
        pass


    def get_random_errs(self) -> dict[str,np.ndarray]:
        pass


    def get_measurements(self) -> dict[str,np.ndarray]:
        pass


    def get_measurement_data(self) -> MeasurementData:
        pass

