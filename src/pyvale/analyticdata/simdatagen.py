'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from dataclasses import dataclass

@dataclass
class AnalyticCaseData:
    length_x: float = 10.0
    length_y: float = 7.5
    num_elem_x: int = 4
    num_elem_y: int = 3


class AnalyticSimDataGen:
    def __init__(self) -> None:
        pass
