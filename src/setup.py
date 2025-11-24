from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "synergy_optimizer",
        ["synergy_optimizer.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="synergy_optimizer",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)