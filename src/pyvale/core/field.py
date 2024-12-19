"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
import mooseherder as mh


class IField(ABC):
    """Interface (abstract base class) for simulation data of physical fields.
    """

    @abstractmethod
    def set_sim_data(self,sim_data: mh.SimData) -> None:
        pass

    @abstractmethod
    def get_sim_data(self) -> mh.SimData:
        pass

    @abstractmethod
    def get_time_steps(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_visualiser(self) -> pv.UnstructuredGrid:
        pass

    @abstractmethod
    def get_all_components(self) -> tuple[str,...]:
        pass

    @abstractmethod
    def get_component_index(self,comp: str) -> int:
        pass

    @abstractmethod
    def sample_field(self,
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    angles: tuple[Rotation,...] | None = None,
                    ) -> np.ndarray:
        pass
