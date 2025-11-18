"""
Setup file for building Cython synergy optimizer.

Build with:
    python setup_synergy.py build_ext --inplace

This compiles the synergy_optimizer.pyx into a C extension for ~10-100x speedup.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "synergy_optimizer",
        ["src/synergy_optimizer.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        extra_link_args=["-O3"],
    )
]

setup(
    name="synergy_optimizer",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
        },
        annotate=True  # Generate HTML annotation files to see Python/C interaction
    ),
    include_dirs=[np.get_include()],
)
