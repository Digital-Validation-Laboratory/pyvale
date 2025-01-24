from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np




extensions = [
    Extension(
        name="cython_interface",
        sources=["cython_interface.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3", "-Wunused-variable"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="RasterMeshBenchmarks",
    ext_modules=cythonize(extensions)
)