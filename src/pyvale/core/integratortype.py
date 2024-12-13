'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import enum

class EIntSpatialType(enum.Enum):
    RECT1PT = enum.auto()
    RECT4PT = enum.auto()
    RECT9PT = enum.auto()
    QUAD4PT = enum.auto()
    QUAD9PT = enum.auto()
