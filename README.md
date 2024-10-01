# pyvale

A python validation engine (pyvale). Used to simulate experimental data from an input multi-physics simulation by explicitly modelling sensors with realistic uncertainties. Useful to test simulation validation metrics and digital shadows/twins.

## Installation
### Virtual Environment

We recommend installing `pyvale` in a virtual environment using `venv` or `pyvale` can be installed into an existing environment of your choice. To create a specific virtual environment for `pyvale` use:

```
python3 -m venv pyvale-env
source pyvale-env/bin/activate
```

### Standard Installation

Clone `pyvale` to your local system and `cd` to the root directory of `pyvale`. Ensure you virtual environment is activated and run from the `pyvale` root directory:

```
pip install .
```

### Developer Installation

To create an editable installation of `pyvale` follow the instructions for a standard installation but run:

```
pip install -e .
```

### `mooseherder`
`pyvale` requires `mooseherder`. go to the `mooseherder` github page (https://github.com/Applied-Materials-Technology/mooseherder) and install `mooseherder` into the same virtual environment as `pyvale`.

## Getting Started

The examples folder includes a sequence of examples using `pyvale`.

## Contributors
The Digital Validation Team at UKAEA:
- Lloyd Fletcher, UK Atomic Energy Authority
- Adel Tayeb, UK Atomic Energy Authority
- Alex Marsh, UK Atomic Energy Authority
- Rory Spencer, UK Atomic Energy Authority
- Michael Atkinson, UK Atomic Energy Authority


