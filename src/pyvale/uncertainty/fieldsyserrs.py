'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''

import numpy as np

from pyvale.physics.field import IField
from pyvale.uncertainty.errorcalculator import IErrCalculator


class SysErrPosition(IErrCalculator):

    def __init__(self, field: IField) -> None:
        self._field = field


    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        return np.array([])


class SysErrSpatialAverage(IErrCalculator):

    def __init__(self, field: IField) -> None:
        self._field = field


    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        return np.array([])


class SysErrTemporalAverage(IErrCalculator):

    def __init__(self, field: IField) -> None:
        self._field = field


    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        return np.array([])