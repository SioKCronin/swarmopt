#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='swarmopt',
    version='0.2.0',  # Major update with new features
    description='Advanced Particle Swarm Optimization with multiobjective support, respect boundaries, and cooperative swarms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/siokcronin/swarmopt',
    author='Siobhan K Cronin',
    author_email='siobhankcronin@gmail.com',
    license='MIT',
    keywords=['particle swarm optimization', 'PSO', 'swarm intelligence', 
              'optimization', 'multiobjective', 'hyperparameter tuning',
              'cooperative PSO', 'NSGA-II', 'machine learning'],
    packages=find_packages(exclude=['tests', 'tests_scripts', 'benchmarks', 
                                   'tutorials']),
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'viz': ['matplotlib>=3.3.0'],
        'multiobjective': ['scipy>=1.5.0'],
        'tda': ['giotto-tda>=0.5.0'],
        'all': ['matplotlib>=3.3.0', 'scipy>=1.5.0'],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/siokcronin/swarmopt/issues',
        'Source': 'https://github.com/siokcronin/swarmopt',
        'Documentation': 'https://github.com/siokcronin/swarmopt/tree/main/docs',
    },
    zip_safe=False,
)
