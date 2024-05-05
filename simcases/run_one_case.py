'''
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import time
from pathlib import Path
from mooseherder import (MooseConfig,
                         MooseRunner)

CASE_STR = 'case05'
CASE_FILES = (CASE_STR+'.geo',CASE_STR+'.i')
CASE_DIR = Path('simcases/'+CASE_STR+'/')

USER_DIR = Path.home()

def main() -> None:
    config = {'main_path': USER_DIR / 'moose',
            'app_path': USER_DIR / 'moose-workdir/proteus',
            'app_name': 'proteus-opt'}

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 7,
                              redirect_out = False)

    input_file = CASE_DIR / CASE_FILES[1]
    print(input_file)

    moose_start_time = time.perf_counter()
    moose_runner.run(input_file)
    moose_run_time = time.perf_counter() - moose_start_time

    print()
    print("="*80)
    print(f'Run time = {moose_run_time:.3f} seconds')
    print("="*80)
    print()


if __name__ == '__main__':
    main()

