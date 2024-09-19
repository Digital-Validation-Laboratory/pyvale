'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pprint import pprint
from typing import Any
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

USER_DIR = Path(str(Path.home()) + '/git/herding-moose/')


def main() -> None:

    OUTPUT_DIR = Path(str(USER_DIR) + '/pyvale/examples/images/')
    
    file_list = os.listdir(OUTPUT_DIR)

    pos_files = list([])
    out_files = list([])
    
    for ff,file in enumerate(file_list):
        if file[:8] == "sens_pos":
            pos_files.append(file)
        if file[:8] == "sim_data":
            out_files.append(file)


    sens_pos = list([])
    sim_data = list([])
    
    for ff in range(len(pos_files)):
        pos_file_path = Path(str(OUTPUT_DIR)+f"/sens_pos_{ff+1}.csv")
        pos_data = pd.read_csv(pos_file_path)
        sens_pos.append(pos_data)
        #print(pos_data)
        out_file_path = Path(str(OUTPUT_DIR)+f"/sim_data_{ff+1}.csv")
        out_data = pd.read_csv(out_file_path)
        sim_data.append(out_data)
        #print(out_data)


if __name__ == '__main__':
    main()