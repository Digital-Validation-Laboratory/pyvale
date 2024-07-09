'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from abc import ABC, abstractmethod

import numpy as np

class ISpatialIntegrator(ABC):
    @abstractmethod
    def calc_averages(self,
                      cent_pos: np.ndarray | None = None,
                      sample_times: np.ndarray | None = None) -> np.ndarray:
        pass

    @abstractmethod
    def get_averages(self) -> np.ndarray:
        pass