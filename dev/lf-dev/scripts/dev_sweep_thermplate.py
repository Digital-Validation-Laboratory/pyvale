'''
================================================================================
pycave: the python computer aided validation engine

License: LGPL-2.1
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
from mooseherder import (MooseHerd,
                         MooseRunner,
                         MooseConfig,
                         InputModifier,
                         DirectoryManager,
                         SweepReader)

USER_DIR = Path.home()
DATA_DIR = Path('dev/lf-dev/')

def main() -> None:
    moose_input = DATA_DIR / 'plate2d_therm_steady.i'
    moose_modifier = InputModifier(moose_input,'#','')

    config = {'main_path': USER_DIR / 'moose',
            'app_path': USER_DIR / 'moose-workdir/proteus',
            'app_name': 'proteus-opt'}
    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 4,
                              redirect_out = True)

    dir_manager = DirectoryManager(n_dirs=4)
    herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)
    herd.set_num_para_sims(n_para=2)

    dir_manager.set_base_dir(DATA_DIR / 'sweep_data/')
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    n_elem_y = [10,20]
    e_mod = [1e9,2e9]
    p_rat = [0.3,0.35]
    moose_vars = list([])
    for nn in n_elem_y:
        for ee in e_mod:
            for pp in p_rat:
                moose_vars.append([{'n_elem_y':nn,'e_modulus':ee,'p_ratio':pp}])

    print(moose_modifier.get_vars())


if __name__ == "__main__":
    main()