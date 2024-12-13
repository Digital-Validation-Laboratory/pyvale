'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
from importlib.resources import files


class DataSet:
    @staticmethod
    def dic_pattern_5mpx_path() -> Path:
        return Path(files("pyvale.data").joinpath("optspeckle_2464x2056px_spec5px_8bit_gblur1px.tiff"))

    @staticmethod
    def thermal_2d_path() -> Path:
        return Path(files("pyvale.data").joinpath("case13_out.e"))

    @staticmethod
    def thermal_3d_path() -> Path:
        return Path(files("pyvale.data").joinpath("case16_out.e"))

    @staticmethod
    def mechanical_2d_path() -> Path:
        return Path(files("pyvale.data").joinpath("case17_out.e"))

    @staticmethod
    def thermomechanical_3d_path() -> Path:
        return Path(files("pyvale.data").joinpath("case18_1_out.e"))

    @staticmethod
    def thermomechanical_3d_experiment_paths() -> list[Path]:
        return [Path(files("pyvale.data").joinpath("case18_1_out.e")),
                Path(files("pyvale.data").joinpath("case18_2_out.e")),
                Path(files("pyvale.data").joinpath("case18_3_out.e"))]

