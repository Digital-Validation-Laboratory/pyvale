"""
==============================================================================
EXAMPLE: Run MOOSE using mooseherder once

Author: Lloyd Fletcher
==============================================================================
"""
import time
from pathlib import Path
from mooseherder import (MooseConfig,
                         MooseRunner)


def main() -> None:
    config = {'main_path': Path.home() / 'moose',
        'app_path': Path.home() / 'proteus',
        'app_name': 'proteus-opt'}

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(n_tasks = 4,
                              n_threads = 2,
                              redirect_out = False)

    input_file = Path('dev/lfdev/meshtest/meshtest_higherorder_2d.i')
    moose_runner.set_input_file(input_file)

    print('Running moose with:')
    print(moose_runner.get_arg_list())

    start_time = time.perf_counter()
    moose_runner.run()
    run_time = time.perf_counter() - start_time

    print()
    print("-"*80)
    print(f'MOOSE run time = {run_time:.3f} seconds')
    print("-"*80)
    print()

if __name__ == '__main__':
    main()

