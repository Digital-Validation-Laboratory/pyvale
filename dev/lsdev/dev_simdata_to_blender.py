from pathlib import Path
import numpy as np
import mooseherder as mh
from dev_partblender import BlenderPart

def main() -> None:
    data_path = Path('src/data/case13_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()




    part = BlenderPart(sim_data).simdata_to_part()

    


if __name__ == '__main__':
    main()




