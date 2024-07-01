
'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from dataclasses import dataclass
import numpy as np

@dataclass
class SensorDescriptor():
    name: str = 'Measured Value'
    units: str = r'-'
    symbol: str = r'm'
    tag: str = 'S'
    components: tuple[str,...] | None = None

    def create_label(self, comp_ind: int | None = None) -> str:
        label = ""
        if self.name != "":
            label = label + rf"{self.name} "

        symbol = rf"${self.symbol}$ "
        if comp_ind is not None and self.components is not None:
            symbol = rf"${self.symbol}_{{{self.components[comp_ind]}}}$ "
        if symbol != "":
            label = label + symbol

        if self.units != "":
            label = label + rf"[${self.units}$]"

        return label

    def create_label_notex(self, comp_ind: int | None = None) -> str:
        label = ""
        if self.name != "":
            label = label + rf"{self.name} "

        symbol = rf"{self.symbol} "
        if comp_ind is not None and self.components is not None:
            symbol = rf"{self.symbol}_{self.components[comp_ind]} "
        if symbol != "":
            label = label + symbol

        if self.units != "":
            label = label + rf"[{self.units}]"

        return label

    def create_sensor_tags(self,n_sensors: int) -> list[str]:
        z_width = int(np.log10(n_sensors))+1

        sensor_names = list()
        for ss in range(n_sensors):
            num_str = f'{ss}'.zfill(z_width)
            sensor_names.append(f'{self.tag}{num_str}')

        return sensor_names

