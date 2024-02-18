'''
-------------------------------------------------------------------------------
pycave: dev_main

authors: thescepticalrabbit
-------------------------------------------------------------------------------
'''
from pprint import pprint
from pathlib import Path
from mooseherder.exodusreader import ExodusReader

def main() -> None:
    data_path = Path('data/moose_thermal_basic_out.e')
    data_reader = ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()


if __name__ == '__main__':
    main()

