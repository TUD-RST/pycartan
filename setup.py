from distutils.core import setup

setup(
    name='pycartan',
    version='0.1.2',
    author='Carsten Knoll, Klemens Fritzsche',
    author_email='Carsten.Knoll@tu-dresden.de',
    packages=['pycartan'],
    url='https://github.com/cknoll/pycartan',
    license='BSD3',
    description='Python library for (vector) differential forms in control theory',
    long_description="""
    Representation of differential forms and vector differential forms
    with respective operations of exterior calculus (exterior
    derivative, wedge product, contraction (interior product)).
    Background is the application differential forms in the context of
    control theory.
    """,
    requires=[
        "sympy (>= 0.7.6)",
        "numpy (>= 1.10.4)",
        "symbtools (>= 0.1.5)",
        "ipython (>= 3.1.0)",
        "ipydex (>= 0.1.0)",
    ],
)
