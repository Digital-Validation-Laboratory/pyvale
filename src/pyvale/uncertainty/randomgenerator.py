'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import numpy as np


class IRandomGenerator(ABC):
    @abstractmethod
    def generate(self, size: tuple[int,...]) -> np.ndarray:
        pass


class NormalGenerator(IRandomGenerator):
    def __init__(self,
                 std: float = 1.0,
                 mean: float = 0.0,
                 seed: int | None = None) -> None:

        self._std =std
        self._mean = mean
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.normal(loc = self._mean,
                                scale = self._std,
                                size = size)


class UniformGenerator(IRandomGenerator):
    def __init__(self,
                 low: float = -1.0,
                 high: float = 1.0,
                 seed: int | None = None) -> None:

        self._low = low
        self._high = high
        self._rng = np.random.default_rng(seed)

    def generate(self, size: tuple[int,...]) -> np.ndarray:
        return self._rng.uniform(low = self._low,
                                 high = self._high,
                                 size = size)