'''
================================================================================
DEV: calibration check

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np

# Known: what the temperature should be - truth
# 1) Calculate the voltage using the true calibration
# 2) Use this voltage to calculate the assumed calibrated temp
# 3) Error = assumed - truth

def assumed_calib(vals: np.ndarray) -> np.ndarray:
    return np.array([])

def truth_calib(vals: np.ndarray) -> np.ndarray:
    return np.array([])

def main() -> None:
    millivolts = np.linspace(0,50,100)


if __name__ == "__main__":
    main()