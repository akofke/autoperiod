#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='autoperiod',
    version='0.1.0',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False,

    install_requires=[
        'numpy',
        'scipy',
        'astropy',
    ],

    tests_require=[
        'pytest',
        'pytest-benchmark'
    ],

    extras_require={
        'plotting': ['matplotlib']
    }
)
