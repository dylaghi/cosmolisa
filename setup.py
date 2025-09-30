import os
import sys
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

lal_prefix = os.environ.get("LAL_PREFIX")
if lal_prefix is None:
    sys.exit(
        "No LAL installation found. Please install LAL from source "
          "or source your LAL installation."
          )

lal_includes = os.path.join(lal_prefix, "include")
lal_libs = os.path.join(lal_prefix, "lib")

common_includes = [numpy.get_include(), lal_includes, "cosmolisa"]
common_libs = ["m", "lal"]
common_libdirs = [lal_libs]
common_compile_args = ['-O3', '-ffast-math']

ext_modules = [
    Extension(
        name="cosmolisa.cosmology",
        sources=["cosmolisa/cosmology.pyx"],
        libraries=common_libs,
        library_dirs=common_libdirs,
        include_dirs=common_includes,
        extra_compile_args=common_compile_args,
    ),
    Extension(
        name="cosmolisa.likelihood",
        sources=["cosmolisa/likelihood.pyx"],
        libraries=common_libs,
        library_dirs=common_libdirs,
        include_dirs=common_includes,
        extra_compile_args=common_compile_args,
    )
]

setup(
    name="cosmolisa",
    description="cosmolisa: a pipeline for cosmological inference.",
    author="Danny Laghi, Walter Del Pozzo",
    author_email="danny.laghi@ligo.org, walter.delpozzo@ligo.org",
    url="https://github.com/dylaghi/cosmolisa",
    license="MIT",
    packages=['cosmolisa'],
    install_requires=['numpy', 'scipy', 'corner', 'cython'],
    ext_modules=cythonize(ext_modules, language_level="3"),
    entry_points={
        'console_scripts': [
            "cosmoLISA = cosmolisa.cosmological_model:main"
            ]
        },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
        ],
    keywords="gravitational waves cosmology bayesian inference",
    package_data={"": ['*.pyx', '*.pxd']},
    )