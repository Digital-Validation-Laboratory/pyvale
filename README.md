# pycave

A python computer aided validation engine (pyCAVE). Used to simulate experimental data from an input multi-physics simulation by explicitly modelling sensors with realistic uncertainties. Useful to test simulation validation metrics and digital shadows/twins.

## Installation
### Virtual Environment

We recommend installing `pycave` in a virtual environment using `venv` or `pycave` can be installed into an existing environment of your choice. To create a specific virtual environment for `pycave` use:

```
python3 -m venv herder-env
source herder-env/bin/activate
```

### Standard Installation

Clone `pycave` to your local system and `cd` to the root directory of `pycave`. Ensure you virtual environment is activated and run from the `pycave` root directory:

```
pip install .
```

### Developer Installation

To create an editable installation of `pycave` follow the instructions for a standard installation but run:

```
pip install -e .
```

### `mooseherder`
`pycave` requires `mooseherder`. go to the `mooseherder` github page (https://github.com/Applied-Materials-Technology/mooseherder) and install `mooseherder` into the same virtual environment as `pycave`.

## Getting Started

The examples folder includes a sequence of examples using `pycave`.

## Contributors

- Lloyd Fletcher, UK Atomic Energy Authority, (TheScepticalRabbit)
