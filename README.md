[![PyPI Package](https://badge.fury.io/py/pycartan.svg)](https://badge.fury.io/py/pycartan)
[![Build Status](https://travis-ci.org/TUD-RST/pycartan.svg?branch=master)](https://travis-ci.org/TUD-RST/pycartan)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.275834.svg)](https://doi.org/10.5281/zenodo.275834)

(English version below)

Allgemeines
===========
Das Paket pycartan enthält Programmcode zur Repräsentation von
Differentialformen und damit zusammenhängender Rechenoperationen (Äußere
Ableitung, Keilprodukt, Kontraktion, Integration, Hodge-Stern).
Hintergrund ist die Anwendung von Differentialformen im Kontext der Regelungstheorie.

Der Programmcode hat den Status von "Forschungscode",
d.h. das Paket befindet sich im Entwicklungszustand.
Trotz, dass wesentliche Teile durch Unittests abgedeckt sind, enthält der Code
mit einer gewissen Wahrscheinlichkeit Fehler.

Eine fragmentarische Dokumentation gibt es als  [Jupyter-Notebook](http://nbviewer.jupyter.org/github/TUD-RST/pycartan/blob/master/doc/pycartan_examples.ipynb).

Feedback-Kontakt: http://cknoll.github.io/pages/impressum.html



General Information
===================
The Package pycartan contains code for representation of differential forms and
respective operations of exterior calculus (exterior derivative, wedge product,
contraction (interior product)). Background is the application differential
forms in the context of control theory.

The package has the status of research-code. Despite significant parts are covered by unittests,
the occurence of bugs (including wrong results) is probable.


A rudimentary documentation is available as  [Jupyter notebook](http://nbviewer.jupyter.org/github/TUD-RST/pycartan/blob/master/doc/pycartan_examples.ipynb).

Feedback-Contact: http://cknoll.github.io/pages/impressum.html

Installation
============
The package pycartan depends on the following python packages:

- sympy
- numpy
- ipython
- symbtools (see https://github.com/TUD-RST/symbtools)
- ipydex (see https://github.com/cknoll/ipydex)

They should be installed automatically if you install pycartan via pip.

Get pycartan using PyPI::

    $ pip install pycartan

or the latest git version::

    $ git clone https://github.com/TUD-RST/pycartan.git

