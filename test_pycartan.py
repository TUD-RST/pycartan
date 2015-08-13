# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 16:43:00 2014

@author: Carsten Knoll (enhancements, tests)
"""

import unittest
import sympy as sp
from sympy import sin, cos, exp

from pycartanlib import pycartan as ct

from IPython import embed as IPS

class ExteriorAlgebraTests(unittest.TestCase):

    def setUp(self):
        self.xx, self.dx = ct.setup_objects("x1, x2, x3, x4, x5")

    def test_basics(self):
        dx1, dx2, dx3, dx4, dx5 = self.dx
        self.assertEquals(dx1.basis, sp.Matrix(self.xx))
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

    def test_wedge_product2(self):
        # here we test the `*` operator which was extended to DifferentialForms
        # after `^`

        dx1, dx2, dx3, dx4, dx5 = self.dx
        self.assertEquals(dx1*dx2, dx1.wp(dx2))
        self.assertEquals(dx5*dx2, - dx2^dx5)
        self.assertEquals((dx5*dx2*dx1).degree, 3)

        # commutativity with scalar functions
        self.assertEquals(dx5*x2*x3*10*dx2*dx1*x1, x2*x3*10*x1*dx5^dx2^dx1)

        zero4_form = ct.DifferentialForm(4, self.xx)
        self.assertEquals((dx5*dx2*dx1*dx5), zero4_form)
        self.assertFalse( any((dx4*dx4).coeff) )

    def test_calculation(self):


        dx1, dx2, dx3, dx4, dx5 = self.dx
        x1, x2, x3, x4, x5 = self.xx

        w1 = x2*dx1 + x5*dx3 - dx4
        w1 += dx2*sp.exp(x3)

        with self.assertRaises(sp.SympifyError) as cm:
            x1.diff(dx1)

        with self.assertRaises(TypeError) as cm:
            x1 + dx2

    def test_calculation2(self):
        dx1, dx2, dx3, dx4, dx5 = self.dx
        x1, x2, x3, x4, x5 = self.xx
        pc = ct
        k, u = sp.symbols('k, u')

        w3 = dx3 - u*dx5
        w4 = dx4 + ( - u*(-k - 1)*cos(x2) ) * dx5

        K = (1+k)*cos(x2)  # Faktor
        q1 = K*w3 + w4

        test_c = q1.c[-1].simplify()  # this should be zero

        self.assertEqual(test_c, 0)

    def test_exterior_derivative(self):
        dx1, dx2, dx3, dx4, dx5 = self.dx
        x1, x2, x3, x4, x5 = self.xx

        w1 = dx1*x1**2 + dx3*x2
        dw1 = w1.d
        self.assertIsInstance(dw1, ct.DifferentialForm)
        self.assertEquals(dw1.degree, 2)
        self.assertEquals(dw1, dx2^dx3)

    def test_rank(self):
        a, l, r = sp.symbols('a, l, r')
        (x1, x2, r), (dx1, dx2, dr) = ct.diffgeo_setup(3)
        aa = a*dr-r*dx1-l*dx2
        self.assertEqual(aa.rank(), 1)

    def test_subs(self):
        a, f, r = sp.symbols('a, f, r')
        (x1, x2, r), (dx1, dx2, dr) = ct.diffgeo_setup(3)
        #aa = a*dr-r*dx1-l*dx2

        w1 = dx1.subs(a, r)
        self.assertEqual(dx1, w1)

        w2 = dx1 + a*dx2 + f(x1, x2)*dr
        w3 = w2.subs(a, r)
        w4 = w2.subs(x1, r)

        sl = zip((a, x1), (r, 3*r+7))
        w5 = w2.subs(sl)
        self.assertEqual(w3, dx1 + r*dx2 + f(x1, x2)*dr)
        self.assertEqual(w4, dx1 + a*dx2 + f(r, x2)*dr)
        self.assertEqual(w5, dx1 + r*dx2 + f(3*r+7, x2)*dr)

    def test_string_representation(self):
        a, f, r = sp.symbols('a, f, r')
        (x1, x2, r), (dx1, dx2, dr) = ct.diffgeo_setup(3)

        s1 = str(dx1)
        self.assertEqual(s1, '(1) dx1')

    def test_simplify(self):
        a, f, r = sp.symbols('a, f, r')
        (x1, x2, x3), (dx1, dx2, dx3) = ct.diffgeo_setup(3)

        w = cos(x3)**2*dx1 + sin(x3)**2*dx1
        w.simplify()

        self.assertEqual(w, dx1)

    def test_expand(self):
        a, f, r = sp.symbols('a, f, r')
        (x1, x2, x3), (dx1, dx2, dx3) = ct.diffgeo_setup(3)

        c = a*(1/a - a) + a**2
        w = c*dx1

        self.assertEqual(w.expand(), dx1)

    def test_pull_back_to_sphere(self):
        """
        pull back the differential of the function F = (y1**2 + y2**2 + y3**2)
        it must vanish on the sphere. Hereby we use spherical coordinates
        to consider the sphere as immersed submanifold of R3
        """

        yy = y1, y2, y3 = sp.symbols("y1:4")
        xx = x1, x2 = sp.symbols("x1:3")

        F = y1**2 + y2**2 + y3**2
        omega = ct.d(F, yy)  # omega = dF

        # spherical coordinates:
        phi = sp.Matrix([cos(x1)*cos(x2),
                         sin(x1)*cos(x2),
                         sin(x2)])

        # calculate the pull back transformation of omega along phi_*
        p = ct.pull_back(phi, xx, omega)

        self.assertTrue(p.is_zero())

    def test_lib_namespace(self):
        # this test was motivated by a bug (diff from sp in ct namespace)
        self.assertFalse('diff' in dir(ct))

    def test_gradient(self):
        x1, x2, x3, x4, x5 = self.xx

        h1 = x1
        h2 = x3*x1*sin(x4)*exp(x5)+x2

        dh1 = ct.d(h1, self.xx)
        dh2 = ct.d(h2, self.xx)

        self.assertEqual(dh1.coeff[0], 1)

        self.assertEqual(dh2.coeff[0], h2.diff(x1))
        self.assertEqual(dh2.coeff[1], h2.diff(x2))
        self.assertEqual(dh2.coeff[2], h2.diff(x3))
        self.assertEqual(dh2.coeff[3], h2.diff(x4))
        self.assertEqual(dh2.coeff[4], h2.diff(x5))

    def test_jet_extend_basis1(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xx_tmp, ddx = ct.setup_objects(xx)

        self.assertTrue(xx is xx_tmp)

        # get the individual forms
        dx1, dx2, dx3 = ddx

        dx1.jet_extend_basis()
        self.assertTrue(len(dx1.basis) == 2*len(xx))
        self.assertTrue(dx1.coeff[0] == 1)
        self.assertFalse( any(dx1.coeff[1:]) )

        # derivative coordinates
        xdot1, xdot2, xdot3 = xxd = ct.st.perform_time_derivative(xx, xx)
        ext_basis = ct.st.row_stack(xx, xxd)

        w1 = xdot1 * dx2
        w1.jet_extend_basis()
        self.assertEqual(w1.basis[3], w1.coeff[1])

        dw1 = w1.d
        res1 = ct.d(xdot1, ext_basis)^ct.d(x2, ext_basis)
        self.assertEqual(dw1, res1)

        w2 = x3*dx2
        self.assertEqual(w2.dim_basis, len(xx))
        w2.jet_extend_basis()
        self.assertEqual(w2.dim_basis, len(ext_basis))

        dw2 = w2.d
        res2 = -dx2^dx3
        res2.jet_extend_basis()
        self.assertEqual(dw2, res2)

        # TODO: test with multiple calls to jet_ext_basis

    def test_dot(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xdot1, xdot2, xdot3 = xxd = ct.st.perform_time_derivative(xx, xx)
        xx_tmp, ddx = ct.setup_objects(xx)

        full_basis = list(xx) + list(xxd)
        dxdot2 = ct.DifferentialForm(1, full_basis, [0,0,0, 0,1,0])

        w1 = x3*dx2
        with self.assertRaises(ValueError) as cm:
            w1.dot()

        w1.jet_extend_basis()
        dx2.jet_extend_basis()
        wdot1 = w1.dot()

        self.assertEqual(wdot1, xdot3*dx2 + x3*dxdot2)

    def test_dot2(self):
        a1, a2, a3 = aa = sp.Matrix(sp.symbols("a1:4"))
        adot1, adot2, adot3 = aad = ct.st.perform_time_derivative(aa, aa)

        xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xxd = ct.st.perform_time_derivative(xx, xx)
        xxdd = ct.st.perform_time_derivative(xx, xx, order=2)

        full_basis = ct.st.row_stack(xx, xxd, xxdd)
        xx_tmp, ddx = ct.setup_objects(full_basis)


        w1 = a1*dx2
        wdot1_1 = w1.dot()

        self.assertEqual(wdot1_1, a1*dxdot2)

        wdot1_2 = w1.dot(aa)
        self.assertEqual(wdot1_2, adot1*dx2 + a1*dxdot2)

        w2 = a1*dx2 + a2*dxdot2
        wdot2_expected = adot1*dx2 + (a1 + adot2)*dxdot2 + a2 * dxddot2

        self.assertEqual(wdot2_expected.coeff, w2.dot(aa).coeff)


def main():
    unittest.main()

if __name__ == '__main__':
    main()