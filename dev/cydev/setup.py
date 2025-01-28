from setuptools import Extension, setup
from Cython.Build import cythonize
import sys

# if sys.platform.startswith("win"):
#     openmp_arg = '/openmp'
# else:
#     openmp_arg = '-fopenmp'

ext_modules = [
    Extension(
        "camerac",
        ["camerac.py"],
        extra_compile_args=["-ffast-math",'-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    ext_modules=cythonize(ext_modules,
                          annotate=True)
)
