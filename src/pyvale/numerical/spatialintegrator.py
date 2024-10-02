'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import enum
from abc import ABC, abstractmethod

import numpy as np


class SpatialIntType(enum.Enum):
    RECT1PT = enum.auto()
    RECT4PT = enum.auto()
    RECT9PT = enum.auto()
    QUAD4PT = enum.auto()
    QUAD9PT = enum.auto()


def create_int_pt_array(int_pt_offsets: np.ndarray,
                        cent_pos: np.ndarray,
                        ) -> np.ndarray:
    offset_array = np.tile(int_pt_offsets,(cent_pos.shape[0],1))
    int_pt_array = np.repeat(cent_pos,int_pt_offsets.shape[0],axis=0)
    # shape=(n_sens*n_int_pts,n_dims)
    return int_pt_array + offset_array


class ISpatialIntegrator(ABC):
    @abstractmethod
    def calc_averages(self,
                      cent_pos: np.ndarray | None = None,
                      sample_times: np.ndarray | None = None) -> np.ndarray:
        pass

    @abstractmethod
    def get_averages(self) -> np.ndarray:
        pass

