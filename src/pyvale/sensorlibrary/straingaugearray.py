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

from pyvale.field import TensorField
from pyvale.sensorarray import SensorArray, MeasurementData
from pyvale.plotprops import PlotProps

# TODO:
class StrainGaugeArray(SensorArray):

    def get_positions(self) -> np.ndarray:
        pass


    def get_sample_times(self) -> np.ndarray:
        pass


    def get_truth_values(self) -> dict[str,np.ndarray]:
        pass


    def get_systematic_errs(self) -> dict[str,np.ndarray]:
        pass


    def get_random_errs(self) -> dict[str,np.ndarray]:
        pass


    def get_measurements(self) -> dict[str,np.ndarray]:
        pass


    def get_measurement_data(self) -> MeasurementData:
        pass

