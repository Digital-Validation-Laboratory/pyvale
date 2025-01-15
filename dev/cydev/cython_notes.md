
## setup.py
```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fibonacci.py"),
)
```

## Build
```shell
python setup.py build_ext --inplace
```

## Checking if a module is compiled
In the module:
```python
import cython
if cython.compiled:
    print("Yep, I'm compiled.")
else:
    print("Just a lowly interpreted script.")
```

In the script running the module look at the __file__ repr:
```python
import MODULE

print(MODULE.__file__)
```
It should be the compiled *.so