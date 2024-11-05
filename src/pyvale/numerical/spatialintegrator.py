'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import numpy as np
from pyvale.sensors.sensordata import SensorData


def create_int_pt_array(sens_data: SensorData,
                        int_pt_offsets: np.ndarray,
                        ) -> np.ndarray:

    # shape=(n_sens*n_int_pts,n_dims)
    offset_array = np.tile(int_pt_offsets,(sens_data.positions.shape[0],1))

    if sens_data.angles is not None:
        for rr in sens_data.angles:
            offset_array[,:] = np.matmul(rr.as_matrix(),int_pt_offsets.T).T

    int_pt_array = np.repeat(sens_data.positions,int_pt_offsets.shape[0],axis=0)
    # shape=(n_sens*n_int_pts,n_dims)
    return int_pt_array + offset_array


class ISpatialIntegrator(ABC):
    @abstractmethod
    def calc_averages(self,
                      sens_data: SensorData) -> np.ndarray:
        pass

    @abstractmethod
    def get_averages(self) -> np.ndarray:
        pass

