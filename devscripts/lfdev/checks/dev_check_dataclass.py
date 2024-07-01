'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from dataclasses import dataclass

@dataclass
class SensorDescriptor():
    name: str = 'Measured Value'
    units: str = r'-'
    symbol: str = r'm'
    tag: str = 'S'
    components: tuple[str,...] = ('x','y','z','xy','yz','xz')

    def create_label_str(self, comp_ind: int | None = None) -> str:
        label = ""
        if self.name != "":
            label = label + rf"{self.name} "

        symbol = rf"${self.symbol}$ "
        if comp_ind is not None:
            symbol = rf"${self.symbol}_{{{self.components[comp_ind]}}}$ "
        if symbol != "":
            label = label + symbol

        if self.units != "":
            label = label + rf"[${self.units}$]"

        return label


descripts = SensorDescriptor()

print(descripts.create_label_str())
print(descripts.create_label_str(4))