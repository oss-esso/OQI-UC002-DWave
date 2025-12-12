"""
Setup script to compile fast_rotation_bqm Cython extension.

Usage:
    python setup_fast_rotation.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "fast_rotation_bqm",
        ["fast_rotation_bqm.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-ffast-math'],
    )
]

setup(
    name="fast_rotation_bqm",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
