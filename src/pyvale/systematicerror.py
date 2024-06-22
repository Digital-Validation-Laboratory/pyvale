'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Callable
import numpy as np

# - Have a list of systematic error functions
#   - Calc = loop over the list and calculate the systematic errors
#   - Calc = loop over components and calculate systematic errors
# - This is going to be computationally expensive...

class SystematicErrorGenerator():
    def __init__(self, err_funcs: list[Callable]) -> None:
        self._sys_errs = dict()
        self._sys_err_funcs = err_funcs

    def set_err_funcs(self,err_funcs: list[Callable]) -> None:
        self._sys_err_funcs = err_funcs

    def add_err_func(self,err_func: Callable) -> None:
        self._sys_err_funcs.append(err_func)

    def calc_sys_errs(self) -> dict[str,np.ndarray] | None:

        self._sys_errs = dict()


    def get_sys_errs(self) -> dict[str,np.ndarray] | None:
        return self._sys_errs






