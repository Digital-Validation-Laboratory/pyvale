"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from abc import ABC, abstractmethod
import numpy as np
from pyvale.core.sensordata import SensorData


def create_int_pt_array(sens_data: SensorData,
                        int_pt_offsets: np.ndarray,
                        ) -> np.ndarray:

    n_sens = sens_data.positions.shape[0]
    n_int_pts = int_pt_offsets.shape[0]

    # shape=(n_sens*n_int_pts,n_dims)
    offset_array = np.tile(int_pt_offsets,(n_sens,1))

    if sens_data.angles is not None:
        for ii,rr in enumerate(sens_data.angles):
            offset_array[ii*n_int_pts:(ii+1)*n_int_pts,:] = \
                np.matmul(rr.as_matrix(),int_pt_offsets.T).T

    # shape=(n_sens*n_int_pts,n_dims)
    int_pt_array = np.repeat(sens_data.positions,int_pt_offsets.shape[0],axis=0)

    return int_pt_array + offset_array


class IIntegratorSpatial(ABC):
    """Interface (abstract base class) for ...
    """
    
    @abstractmethod
    def calc_averages(self, sens_data: SensorData) -> np.ndarray:
        pass

    @abstractmethod
    def get_averages(self) -> np.ndarray:
        pass

