#!/usr/bin/env python

from setuptools import setup, find_packages

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
    ],

    tests_require=[
        'pytest',
    ]
)
