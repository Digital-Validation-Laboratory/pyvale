import h5py
from pathlib import Path

file = h5py.File(Path().home() / 'Test001_19-0kW' / 'Test001_19-0kW.H5', 'r')

print(f"{file["DIC Data"]["Listed Data"].keys()=}")
print(f"{file["DIC Data"]["Listed Data"]["X"].shape}")
