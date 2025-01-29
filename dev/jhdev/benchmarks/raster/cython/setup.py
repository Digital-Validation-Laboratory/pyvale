from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="cython_interface",
        sources=["/home/kc4736/ukaea/pyvale/dev/jhdev/rastermeshbenchmarks/cython/cython_interface.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3", "-Wunused-variable"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="RasterMeshBenchmarks",
    ext_modules=cythonize(extensions)
)

#### build command ####
# python3 dev/jhdev/rastermeshbenchmarks/cython/setup.py build_ext --build-lib=dev/jhdev/rastermeshbenchmarks/cython/