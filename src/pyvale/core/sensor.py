'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import numpy as np
from pyvale.core.field import IField


class ISensor(ABC):
    @abstractmethod
    def get_measurement_shape(self) -> tuple[int,int,int]:
        pass

    @abstractmethod
    def get_field(self) -> IField:
        pass

    @abstractmethod
    def calc_measurements(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_measurements(self) -> np.ndarray:
        pass

