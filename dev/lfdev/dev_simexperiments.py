'''
================================================================================
Analytic test case data - linear

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyvale
import mooseherder as mh

""" TODO
An experiment should be able to:
- Perturb simulation input paramaters
    - Some might need to be sampled from a p-dist
    - Some as grid search
- Run simulation sweep

- Apply sensor array to each simulation combination
    - Generate 'N' experiments for each combination

- Plot error intervals for sensor traces

- Pickle the data for later use

An experiment has:
- A simulation / simulation workflow manager
    - The variables to perturb and how to perturb them
- A series of sensors of different types
- Simulation variables to analyse
- The number of experiments to run for each simulation
 """

@dataclass
class ExperimentConfig:
    num_exp_per_sim: int = 30
    variables: dict[str,(float,float)] | None = None


class ExperimentSimulator:
    def __init__(self,
                 sim_data_list,
                 sim_vars: dict[str,(float,...)],
                 ) -> None:

        self.sim_vars = sim_vars

    def run_experiments(self) -> None:
        pass



def main() -> None:
    pass


if __name__ == "__main__":
    main()