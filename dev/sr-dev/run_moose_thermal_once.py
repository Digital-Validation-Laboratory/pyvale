'''
================================================================================
pycave: the python computer aided validation engine.
license: LGPL-2.1
Copyright (C) 2024 Lloyd Fletcher (scepticalrabbit)
================================================================================
'''

import os
from pathlib import Path
from mooseherder.mooseconfig import MooseConfig
from mooseherder.mooserunner import MooseRunner

USER_DIR = Path.home()

def main() -> None:
    config = {'main_path': USER_DIR / 'moose',
            'app_path': USER_DIR / 'moose-workdir/proteus',
            'app_name': 'proteus-opt'}

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(n_tasks = 1, n_threads = 6, redirect_out = False)


    #input_file = Path('data/plate_2d_thermal.i')
    input_file = Path('data/monoblock_3d_thermal.i')
    moose_runner.run(input_file)

if __name__ == '__main__':
    main()


