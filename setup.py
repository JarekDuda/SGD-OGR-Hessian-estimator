from setuptools import setup, find_packages

setup(
    name='sgd-ogr',
    version='0.0.1',
    packages=find_packages(include=('sgd_ogr',), exclude=('tests',)),
)