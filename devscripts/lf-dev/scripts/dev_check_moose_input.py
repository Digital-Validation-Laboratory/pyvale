"""
==============================================================================
EXAMPLE: Modify input script for MOOSE with mooseherder

Author: Lloyd Fletcher
==============================================================================
"""
from pprint import pprint
from pathlib import Path
from mooseherder import InputModifier

DATA_PATH = Path('dev/lf-dev')
MOOSE_INPUT = DATA_PATH / 'plate2d_therm_steady.i'
MOOSE_SAVE = DATA_PATH / 'mod_check.i'

def main() -> None:
    """main: modify moose input script and write modified file.
    """
    print("-"*80)
    print("Modify MOOSE input script")
    print("-"*80)
    moose_mod = InputModifier(MOOSE_INPUT, comment_char="#", end_char="")

    print("Variables found the top of the MOOSE input file:")
    pprint(moose_mod.get_vars())
    print()

    print("New variables inserted:")
    print(moose_mod.get_vars())
    print()

    moose_mod.write_file(MOOSE_SAVE)

    print("Modified input script written to:")
    print(MOOSE_SAVE)
    print()

    print("Example complete.")
    print("-"*80)
    print()


if __name__ == "__main__":
    main()
