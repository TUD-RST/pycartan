# -*- coding: utf-8 -*-

from setuptools import setup
from pycartan.release import __version__

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read()

setup(
    name='pycartan',
    version=__version__,
    author='Carsten Knoll, Klemens Fritzsche',
    author_email='Carsten.Knoll@tu-dresden.de',
    packages=['pycartan'],
    url='https://github.com/TUD-RST/pycartan',
    license='BSD3',
    description='Python library for (vector) differential forms in control theory',
    long_description="""
    Representation of differential forms and vector differential forms
    with respective operations of exterior calculus (exterior
    derivative, wedge product, contraction (interior product), integration, hodge-star).
    Background is the application differential forms in the context of
    control theory.
    """,
    keywords='differential forms, wedge product, exterior derivative, frobenius theorem, differential flat systems',
    install_requires=requirements,
)
