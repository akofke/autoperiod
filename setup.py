#!/usr/bin/env python

from distutils.core import setup

from setuptools import find_packages

setup(
    name='autoperiod',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False,

    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib'
    ]
)