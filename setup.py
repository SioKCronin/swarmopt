from setuptools import setup, find_packages
import sys

if sys.version_info.major !=3:
    print('This application is only compatible with Python 3, but you are running '
          'Python {}. The installation may fail.'.format(sys.version_info.major))

setup(name='PSO-papers',
      packages=[package for package in find_packages()
                if package.starswith('PSO-baselines')],
      install_requires=[
          'pytorch',
          'pyswarms',
          'tensorflow>=1.4.0'],
      description='Baseline implementation of PSO variations',
      author='SioKCronin',
      url='https://github.com/SioKCronin/PSO-baselines',
      version='0.1')

