'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from dataclasses import dataclass

import numpy as np

@dataclass
class MeasurementData():
    measurements: np.ndarray | None =  None
    random_errs: np.ndarray | None  = None
    systematic_errs: np.ndarray | None = None
    truth_values: np.ndarray | None = None


'''
def calc_measurement_data(self) -> MeasurementData:
    measurement_data = MeasurementData()
    measurement_data.measurements = self.calc_measurements()
    measurement_data.systematic_errs = self.get_systematic_errs()
    measurement_data.random_errs = self.get_random_errs()
    measurement_data.truth_values = self.get_truth_values()
    return measurement_data

def get_measurement_data(self) -> MeasurementData:
    measurement_data = MeasurementData()
    measurement_data.measurements = self.get_measurements()
    measurement_data.systematic_errs = self.get_systematic_errs()
    measurement_data.random_errs = self.get_random_errs()
    measurement_data.truth_values = self.get_truth_values()
    return measurement_data
'''