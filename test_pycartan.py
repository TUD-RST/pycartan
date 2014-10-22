# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 16:43:00 2014

@author: Carsten Knoll (enhancements, tests)
"""

import unittest
import sympy as sp
from sympy import sin, cos

import diffgeopy as ct

class ExteriorAlgebraTests(unittest.TestCase):

    def setUp(self):
        self.xx, self.dx = ct.diffgeo_setup("x1, x2, x3, x4, x5")

    def test_basics(self):
        dx1, dx2, dx3, dx4, dx5 = self.dx
        self.assertEquals(dx1.basis, self.xx)
        self.assertEquals(dx1.degree, 1)
        # TODO: add some more basics

    def test_wedge_product1(self):

        dx1, dx2, dx3, dx4, dx5 = self.dx
        self.assertEquals(dx1^dx2, dx1.wp(dx2))
        self.assertEquals(dx5^dx2, - dx2^dx5)
        self.assertEquals((dx5^dx2^dx1).degree, 3)
        zero4_form = ct.DifferentialForm(4, self.xx)
        self.assertEquals((dx5^dx2^dx1^dx5), zero4_form)
        self.assertFalse( any((dx4^dx4).coeff) )

    def test_calculation(self):

        dx1, dx2, dx3, dx4, dx5 = self.dx
        x1, x2, x3, x4, x5 = self.xx


        w1 = x2*dx1 + x5*dx3 - dx4
        w1 += dx2* sp.exp(x3)

        self.assertIsInstance(w1, ct.DifferentialForm)
        self.assertEquals(w1.degree, 1)

    def test_exterior_derivative(self):
        dx1, dx2, dx3, dx4, dx5 = self.dx
        x1, x2, x3, x4, x5 = self.xx

        w1 = dx1*x1**2 + dx3*x2
        dw1 = w1.d
        self.assertIsInstance(dw1, ct.DifferentialForm)
        self.assertEquals(dw1.degree, 2)
        self.assertEquals(dw1, dx2^dx3)

    def test_pull_back_to_sphere(self):
        """
        pull back the differential of the function F = (y1**2 + y2**2 + y3**2)
        it must vanish on the sphere. Hereby we use spherical coordinates
        to consider the sphere as immersed submanifold of R3
        """

        yy = y1, y2, y3 = sp.symbols("y1:4")
        xx = x1, x2 = sp.symbols("x1:3")

        F = y1**2 + y2**2 + y3**2
        omega = dF = ct.d(F, yy)

        # spherical coordinates:
        phi = sp.Matrix([cos(x1)*cos(x2),
                         sin(x1)*cos(x2),
                         sin(x2)])

        # calculate the pull back transformation of omega along phi_*
        p = ct.pull_back(phi, xx, omega)

        self.assertTrue(p.is_zero())



def main():
    unittest.main()

if __name__ == '__main__':
    main()