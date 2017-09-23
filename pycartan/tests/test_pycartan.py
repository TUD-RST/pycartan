# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 16:43:00 2014

@author: Carsten Knoll
@author: Klemens Fritzsche
"""

import unittest
import sympy as sp
from sympy import sin, cos, exp, tan

import pycartan as pc
import symbtools as st
import symbtools.noncommutativetools as nct

from ipydex import IPS


# noinspection PyUnresolvedReferences
# noinspection PyPep8Naming
class ExteriorAlgebraTests(unittest.TestCase):

    def setUp(self):
        self.xx, self.dx = pc.setup_objects("x1, x2, x3, x4, x5")

    def test_basics(self):
        dx1, dx2, dx3, dx4, dx5 = self.dx
        self.assertEqual(dx1.basis, sp.Matrix(self.xx))
        self.assertEqual(dx1.degree, 1)
        # TODO: add some more basics

    def test_wedge_product1(self):

        dx1, dx2, dx3, dx4, dx5 = self.dx
        self.assertEqual(dx1^dx2, dx1.wp(dx2))
        self.assertEqual(dx5^dx2, - dx2^dx5)
        self.assertEqual((dx5^dx2^dx1).degree, 3)
        zero4_form = pc.DifferentialForm(4, self.xx)
        self.assertEqual((dx5^dx2^dx1^dx5), zero4_form)
        self.assertFalse( any((dx4^dx4).coeff) )

    def test_wedge_product2(self):
        # here we test the `*` operator which was extended to DifferentialForms
        # after `^`

        x1, x2, x3, x4, x5 = self.xx
        dx1, dx2, dx3, dx4, dx5 = self.dx
        self.assertEqual(dx1*dx2, dx1.wp(dx2))
        self.assertEqual(dx5*dx2, - dx2^dx5)
        self.assertEqual((dx5*dx2*dx1).degree, 3)

        # commutativity with scalar functions
        self.assertEqual(dx5*x2*x3*10*dx2*dx1*x1, x2*x3*10*x1*dx5^dx2^dx1)

        zero4_form = pc.DifferentialForm(4, self.xx)
        self.assertEqual((dx5*dx2*dx1*dx5), zero4_form)
        self.assertFalse( any((dx4*dx4).coeff) )

    def test_calculation(self):


        dx1, dx2, dx3, dx4, dx5 = self.dx
        x1, x2, x3, x4, x5 = self.xx

        w1 = x2*dx1 + x5*dx3 - dx4
        w1 += dx2*sp.exp(x3)

        w2 = w1/x3

        w3 = w1*1/x1
        w4 = w1/2
        w5 = w1/2.1

        with self.assertRaises(sp.SympifyError) as cm:
            x1.diff(dx1)

        with self.assertRaises(TypeError) as cm:
            x1 + dx2

        with self.assertRaises(TypeError) as cm:
            1/dx2

        with self.assertRaises(TypeError) as cm:
            dx1/dx2

        with self.assertRaises(Exception) as cm:
            # raises NotImplemented but this might change in further versions of sympy
            dx1/sp.eye(3)

    def test_calculation_a(self):
        xx = st.symb_vector('x1:6')

        dx1 = pc.DifferentialForm(1, xx, coeff=[1,0,0,0,0])

    def test_calculation2(self):
        dx1, dx2, dx3, dx4, dx5 = self.dx
        x1, x2, x3, x4, x5 = self.xx
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
        self.assertIsInstance(dw1, pc.DifferentialForm)
        self.assertEqual(dw1.degree, 2)
        self.assertEqual(dw1, dx2^dx3)

    def test_rank(self):
        a, l, r = sp.symbols('a, l, r')
        (x1, x2, r), (dx1, dx2, dr) = pc.diffgeo_setup(3)
        aa = a*dr-r*dx1-l*dx2
        self.assertEqual(aa.rank(), 1)

    def test_subs(self):
        a, f, r = sp.symbols('a, f, r')
        (x1, x2, r), (dx1, dx2, dr) = pc.diffgeo_setup(3)
        #aa = a*dr-r*dx1-l*dx2

        w1 = dx1.subs(a, r)
        self.assertEqual(dx1, w1)

        w2 = dx1 + a*dx2 + f(x1, x2)*dr
        w3 = w2.subs(a, r)
        w4 = w2.subs(x1, r)

        sl = list(zip((a, x1), (r, 3*r+7)))
        w5 = w2.subs(sl)
        self.assertEqual(w3, dx1 + r*dx2 + f(x1, x2)*dr)
        self.assertEqual(w4, dx1 + a*dx2 + f(r, x2)*dr)
        self.assertEqual(w5, dx1 + r*dx2 + f(3*r+7, x2)*dr)

    def test_string_representation(self):
        a, f, r = sp.symbols('a, f, r')
        (x1, x2, x3), (dx1, dx2, dx3) = pc.diffgeo_setup(3)

        s1 = str(dx1)
        self.assertEqual(s1, '(1)dx1')

        s1 = str(0*dx2)
        self.assertEqual(s1, '(0)dx1')

        s1 = str(7*a*dx1*dx2 - dx2*dx3)
        self.assertEqual(s1, '(7*a)dx1^dx2  +  (-1)dx2^dx3')

        w2 = -sin(x3)*dx1 + cos(x3)*dx2
        w3 = dx3
        s1 = str(w2.d^w3)
        self.assertEqual(s1, '(0)dx1^dx2^dx3')

        s1 = str(w2.d^w2^w3)
        self.assertEqual(s1, '(0)dx1^dx1^dx1^dx1')

        s1 = str(w2^w2^w2^ w2^w2^w2)
        self.assertEqual(s1, '(0)dx1'+'^dx1'*5)



    def test_simplify(self):
        a, f, r = sp.symbols('a, f, r')
        (x1, x2, x3), (dx1, dx2, dx3) = pc.diffgeo_setup(3)

        w = cos(x3)**2*dx1 + sin(x3)**2*dx1
        w.simplify()

        self.assertEqual(w, dx1)

    def test_expand(self):
        a, f, r = sp.symbols('a, f, r')
        (x1, x2, x3), (dx1, dx2, dx3) = pc.diffgeo_setup(3)

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
        omega = pc.d(F, yy)  # omega = dF

        # spherical coordinates:
        phi = sp.Matrix([cos(x1)*cos(x2),
                         sin(x1)*cos(x2),
                         sin(x2)])

        # calculate the pull back transformation of omega along phi_*
        p = pc.pull_back(phi, xx, omega)

        self.assertTrue(p.is_zero())

    def test_lib_namespace(self):
        # this test was motivated by a bug (diff from sp in ct namespace)
        self.assertFalse('diff' in dir(pc))

    def test_gradient(self):
        x1, x2, x3, x4, x5 = self.xx

        h1 = x1
        h2 = x3*x1*sin(x4)*exp(x5)+x2

        dh1 = pc.d(h1, self.xx)
        dh2 = pc.d(h2, self.xx)

        self.assertEqual(dh1.coeff[0], 1)

        self.assertEqual(dh2.coeff[0], h2.diff(x1))
        self.assertEqual(dh2.coeff[1], h2.diff(x2))
        self.assertEqual(dh2.coeff[2], h2.diff(x3))
        self.assertEqual(dh2.coeff[3], h2.diff(x4))
        self.assertEqual(dh2.coeff[4], h2.diff(x5))

    def test_integrate1(self):
        x1, x2, x3, x4, x5 = self.xx

        a, b = sp.symbols('a, b', nonzero=True)

        if 1:
            y1 = x1 + sin(x3)*x2
            dy1 = pc.d(y1, self.xx)
            self.assertTrue(dy1.d.is_zero())
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)

            y1 = 0
            dy1 = pc.d(y1, self.xx)
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)

            y1 = x1
            dy1 = pc.d(y1, self.xx)
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)

            y1 = x1+x2
            dy1 = pc.d(y1, self.xx)
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)

            y1 = x2*cos(x1) + x5*cos(x4)
            dy1 = pc.d(y1, self.xx)
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)

            y1 = sin(x1)*cos(x2) + x5*cos(x4)
            dy1 = pc.d(y1, self.xx)
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)

            y1 = sin(x1 + x2 + x3)
            dy1 = pc.d(y1, self.xx)
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)

            y1 = a*x1 + b*x2 + a*b*cos(x3)
            dy1 = pc.d(y1, self.xx)
            y1b = dy1.integrate()
            self.assertEqual(sp.simplify(y1 - y1b), 0)

            y1 = x1 + sin(x3)*x2**2*sp.exp(x1) + sin(x2)*x3
            # y1 = x1 + sin(x3)*x2**1*sp.exp(x1) + sin(x2)*x3
            dy1 = pc.d(y1, self.xx)
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)

            y1 = sin(x1 + x2 + x3)
            dy1 = pc.d(y1, self.xx)
            w = x1*dy1
            self.assertFalse(w.d.is_zero())
            with self.assertRaises(ValueError) as cm:
                w.integrate()

    def test_integrate2(self):
        x1, x2, x3, x4, x5 = self.xx

        a, b = sp.symbols('a, b', nonzero=True)

        if 1:
            # tests for some simplification problem

            y1 = sp.log(x2) + sp.log(cos(x1))
            dx1, dx2, dx3, dx4, dx5 = self.dx
            dy1 = (-sp.tan(x1))*dx1 + (1/x2)*dx2
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)

            y1 = sp.log(x2) + sp.log(sin(x1) - 1)/2 + sp.log(sin(x1) + 1)/2
            dx1, dx2, dx3, dx4, dx5 = self.dx
            dy1 = (-sp.tan(x1))*dx1 + (1/x2)*dx2
            y1b = dy1.integrate()
            difference = y1-y1b
            # this is not zero but it does not depend on xx:
            grad = sp.simplify(st.gradient(difference, self.xx))
            self.assertEqual(grad, grad*0)

            # another variant
            y1 = sp.log(x2) + sp.log(sin(x1) - 1)/2 + sp.log(sin(x1) + 1)/2
            dx1, dx2, dx3, dx4, dx5 = self.dx
            dy1 = (-sp.sin(x1)/sp.cos(x1))*dx1 + (1/x2)*dx2
            y1b = dy1.integrate()
            difference = y1-y1b
            # this is not zero but it does not depend on xx:
            grad = sp.simplify(st.gradient(difference, self.xx))
            self.assertEqual(grad, grad*0)

            y1 = a*sp.log(cos(b*x1))
            dy1 = pc.d(y1, self.xx)
            y1b = dy1.integrate()
            self.assertEqual(sp.simplify(y1 - y1b), 0)


    def test_integrate3(self):
        x1, x2, x3, x4, x5 = self.xx

        a, b = sp.symbols('a, b', nonzero=True)

        if 1:
            y1 = -x3*cos(x1)
            dy1 = pc.d(y1, self.xx)
            print(dy1)
            self.assertTrue(dy1.d.is_zero())
            y1b = dy1.integrate()
            self.assertEqual(y1, y1b)


    def test_jet_extend_basis1(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xx_tmp, ddx = pc.setup_objects(xx)

        self.assertTrue(xx is xx_tmp)

        # get the individual forms
        dx1, dx2, dx3 = ddx

        dx1.jet_extend_basis()
        self.assertTrue(len(dx1.basis) == 2*len(xx))
        self.assertTrue(dx1.coeff[0] == 1)
        self.assertFalse( any(dx1.coeff[1:]) )

        # derivative coordinates
        xdot1, xdot2, xdot3 = xxd = pc.st.time_deriv(xx, xx)
        ext_basis = pc.st.row_stack(xx, xxd)

        w1 = xdot1 * dx2
        w1.jet_extend_basis()
        self.assertEqual(w1.basis[3], w1.coeff[1])

        dw1 = w1.d
        res1 = pc.d(xdot1, ext_basis)^pc.d(x2, ext_basis)
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

    def test_jet_extend_basis1(self):
        x1, x2, x3 = xx = st.symb_vector("x1, x2, x3")
        xx_tmp, ddx = pc.setup_objects(xx)

        self.assertTrue(xx is xx_tmp)

        # get the individual forms
        dx1, dx2, dx3 = ddx

        dx1.jet_extend_basis()
        xdot1, xdot2, xdot3 = xxd = pc.st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxdd = pc.st.time_deriv(xx, xx, order=2)

        full_basis = st.row_stack(xx, xxd, xxdd)

        foo, ddX = pc.setup_objects(full_basis)

        dx1.jet_extend_basis()
        self.assertEqual(ddX[0].basis, dx1.basis)
        self.assertEqual(ddX[0].coeff, dx1.coeff)

        half_basis = st.row_stack(xx, xxd)
        foo, ddY = pc.setup_objects(half_basis)

        dx2.jet_extend_basis()
        self.assertEqual(ddY[1].basis, dx2.basis)
        self.assertEqual(ddY[1].coeff, dx2.coeff)

    def test_dot(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xdot1, xdot2, xdot3 = xxd = pc.st.time_deriv(xx, xx)
        xx_tmp, ddx = pc.setup_objects(xx)
        dx1, dx2, dx3 = ddx

        full_basis = list(xx) + list(xxd)
        dxdot2 = pc.DifferentialForm(1, full_basis, [0,0,0, 0,1,0])

        w1 = x3*dx2
        with self.assertRaises(ValueError) as cm:
            w1.dot()

        w1.jet_extend_basis()
        dx2.jet_extend_basis()
        wdot1 = w1.dot()

        self.assertEqual(wdot1, xdot3*dx2 + x3*dxdot2)

    def test_dot2(self):
        a1, a2, a3 = aa = sp.Matrix(sp.symbols("a1:4"))
        adot1, adot2, adot3 = aad = pc.st.time_deriv(aa, aa)

        xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xxd = pc.st.time_deriv(xx, xx)
        xxdd = pc.st.time_deriv(xx, xx, order=2)

        full_basis = pc.st.row_stack(xx, xxd, xxdd)
        xx_tmp, ddx = pc.setup_objects(full_basis)
        dx1, dx2, dx3, dxdot1, dxdot2, dxdot3, dxddot1, dxddot2, dxddot3 = ddx


        w1 = a1*dx2
        wdot1_1 = w1.dot()

        self.assertEqual(wdot1_1, a1*dxdot2)

        wdot1_2 = w1.dot(aa)
        self.assertEqual(wdot1_2, adot1*dx2 + a1*dxdot2)

        w2 = a1*dx2 + a2*dxdot2
        wdot2_expected = adot1*dx2 + (a1 + adot2)*dxdot2 + a2 * dxddot2

        self.assertEqual(wdot2_expected.coeff, w2.dot(aa).coeff)

    def test_dot3(self):
        xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xxd = pc.st.time_deriv(xx, xx)
        xxdd = pc.st.time_deriv(xx, xx, order=2)


        full_basis = pc.st.row_stack(xx, xxd, xxdd)
        foo, ddx = pc.setup_objects(full_basis)
        dx1, dx2, dx3, dxdot1, dxdot2, dxdot3, dxddot1, dxddot2, dxddot3 = ddx

        aa = sp.Matrix(sp.symbols("a1:3"))
        a1, a2 = aa

        mu1 = a1*dxdot1 + 3*a2*dx2

        mu1.dot()
        # this once was a bug:
        self.assertEqual(xxdd[0].difforder, 2)

    def test_dot_2form(self):
        xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xxd = pc.st.time_deriv(xx, xx)
        xxdd = pc.st.time_deriv(xx, xx, order=2)

        XX = pc.st.row_stack(xx, xxd, xxdd)
        foo, ddx = pc.setup_objects(XX)
        dx1, dx2, dx3, dxdot1, dxdot2, dxdot3, dxddot1, dxddot2, dxddot3 = ddx

        a1, a2 = aa = sp.Matrix(sp.symbols("a1:3"))
        adot1, adot2 = aadot = st.time_deriv(aa, aa)
        b1, b2 = bb = sp.Matrix(sp.symbols("b1:3"))

        z = dx1^dx2
        res = z.dot()
        # Caution: using * instead of ^ because of pythons operator precedence (+ before ^)
        self.assertEqual(res, dxdot1*dx2 + dx1*dxdot2)

        z = 0*dx1^dx2
        res = z.dot()
        self.assertEqual(res, z)

        z = a1*dx1^dx2
        res = z.dot(additional_symbols=aa)
        self.assertEqual(res, adot1*dx1*dx2 + a1*dxdot1*dx2 + a1*dx1*dxdot2)

        z = a1*dxdot1^dx2
        res = z.dot(additional_symbols=aa)
        self.assertEqual(res, adot1*dxdot1*dx2 + a1*dxddot1*dx2 + a1*dxdot1*dxdot2)

        z = a1*dx1*dx2 + a2*dxdot1*dx2
        res = z.dot(additional_symbols=aa)
        exp_res = adot1*dx1*dx2 + (a1 + adot2)*dxdot1*dx2 + a1*dx1*dxdot2 + a2*dxdot1*dxdot2 +\
                  a2*dxddot1*dx2
        self.assertEqual(res, exp_res)

    def test_ord(self):
        x1, x2, x3 = xx = st.symb_vector("x1, x2, x3")
        xdot1, xdot2, xdot3 = xxd = pc.st.time_deriv(xx, xx)
        xxdd = pc.st.time_deriv(xx, xx, order=2)

        XX = st.concat_rows(xx, xxd, xxdd)
        XX, dXX = pc.setup_objects(XX)


        dx1, dx2, dx3, dxdot1, dxdot2, dxdot3, dxddot1, dxddot2, dxddot3 = dXX

        w0 = 0*dx1
        w1 = dx1 + dxdot3
        w2 = 4*x2*dx1 - sp.sin(x3)*xdot1*dx2


        self.assertEqual(w0.ord, 0)
        self.assertEqual(dx1.ord, 0)
        self.assertEqual(dxdot1.ord, 1)
        self.assertEqual(dxddot3.ord, 2)
        self.assertEqual(w1.ord, 1)
        self.assertEqual(w2.ord, 0)
        self.assertEqual(w2.d.ord, 1)

        w3 = w1^w2

        self.assertEqual(w3.ord, 1)
        self.assertEqual(w3.dot().ord, 2)


    def test_get_coeff(self):
        xx = st.symb_vector("x, y, z")
        (x, y, z), (dx, dy, dz) = pc.setup_objects(xx)
        w = 7*dx - x**2*dy
        c1 = w.get_coeff(dx)
        c2 = w.get_coeff(dy)

        self.assertEqual(c1, 7)
        self.assertEqual(c2, -x**2)

    def test_get_multiplied_baseform(self):

        x1, x2, x3 = xx = st.symb_vector("x1, x2, x3")
        xx, dxx = pc.setup_objects(xx)
        dx1, dx2, dx3 = dxx

        W = 7*(dx1^dx2) + 3*x2*(dx1^dx3)

        res1 = W.get_multiplied_baseform(dx1^dx2)
        self.assertEqual(res1, 7*dx1^dx2)

        res2 = W.get_multiplied_baseform(dx2^dx3)
        self.assertEqual(res2, 0*dx1^dx2)

        res3 = W.get_multiplied_baseform(dx1^dx3)
        self.assertEqual(res3, 3*x2*dx1^dx3)

        res4 = W.get_multiplied_baseform((0, 1))
        self.assertEqual(res4, 7*dx1^dx2)

        res5 = W.get_multiplied_baseform((0, 2))
        self.assertEqual(res5, 3*x2*dx1^dx3)

        res6 = W.get_multiplied_baseform((1, 2))
        self.assertEqual(res6, 0*dx1^dx3)

        with self.assertRaises(ValueError) as cm:
            res = W.get_multiplied_baseform(dx1)

        with self.assertRaises(ValueError) as cm:
            res = W.get_multiplied_baseform(x2*dx1^dx2)

        with self.assertRaises(ValueError) as cm:
            res = W.get_multiplied_baseform(dx2^dx1)

        with self.assertRaises(ValueError) as cm:
            res = W.get_multiplied_baseform((1, 0))

    def test_get_baseform(self):

        x1, x2, x3 = xx = st.symb_vector("x1, x2, x3")
        xx, dxx = pc.setup_objects(xx)
        dx1, dx2, dx3 = dxx

        W = 7*(dx1^dx2) + 3*x2*(dx1^dx3)

        res1 = W.get_baseform_from_idcs((0, 1))
        self.assertEqual(res1, dx1^dx2)

        idcs_matrix = sp.Matrix([0, 2])
        res2 = W.get_baseform_from_idcs(idcs_matrix)
        self.assertEqual(res2, dx1^dx3)

        idcs_array = st.np.array([0, 2])
        res2b = W.get_baseform_from_idcs(idcs_array)
        self.assertEqual(res2b, dx1^dx3)

        res3 = W.get_baseform_from_idcs((1, 2))
        self.assertEqual(res3, dx2^dx3)

        Z = dx1 + x3**2*dx2

        res4 = Z.get_baseform_from_idcs((1, ))
        self.assertEqual(res4, dx2)

        res5 = Z.get_baseform_from_idcs(2)
        self.assertEqual(res5, dx3)

        with self.assertRaises(TypeError) as cm:
            res = W.get_baseform_from_idcs(dx1)

        with self.assertRaises(ValueError) as cm:
            res = W.get_baseform_from_idcs((0, 0))

    def test_get_baseform_from_plain_index(self):
        x1, x2, x3 = xx = st.symb_vector("x1, x2, x3")
        xx, dxx = pc.setup_objects(xx)
        dx1, dx2, dx3 = dxx

        W = 7*(dx1^dx2) + 3*x2*(dx1^dx3)

        res = W.get_baseform_from_plain_index(0)
        self.assertEqual(res, dx1^dx2)

        res = W.get_baseform_from_plain_index(2)
        self.assertEqual(res, dx2^dx3)

        res = W.get_baseform_from_plain_index(-1)
        self.assertEqual(res, dx2^dx3)

        res = W.get_baseform_from_plain_index(-2)
        self.assertEqual(res, dx1^dx3)

        res = W.get_baseform_from_plain_index(-3)
        self.assertEqual(res, dx1^dx2)


        with self.assertRaises(ValueError) as cm:
            res = W.get_baseform_from_plain_index(3)

        with self.assertRaises(ValueError) as cm:
            res = W.get_baseform_from_plain_index(-4)


    def test_coeff_ido_do(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xxd = pc.st.time_deriv(xx, xx)
        xxdd = pc.st.time_deriv(xx, xx, order=2)


        full_basis = pc.st.row_stack(xx, xxd, xxdd)
        foo, ddx = pc.setup_objects(full_basis)
        dx1, dx2, dx3, dxdot1, dxdot2, dxdot3, dxddot1, dxddot2, dxddot3 = ddx

        aa = sp.Matrix(sp.symbols("a1:10"))
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = aa
        aad = pc.st.time_deriv(aa, aa)
        #adot1, adot2, adot3 =\

        mu1 = a1*dxdot1 + 3*a2*dx2
        mu2 = a3*dxdot2 - a4*dx1 + 6*a5*dx2
        mu3 = 7*a6*x1*dxdot3 + a7*dx3

        Mu_0 = mu1^mu2^mu3
        Mu_1 = mu1.dot(aa)^mu2.dot(aa)^mu3.dot(aa)

        # this once was a bug:
        self.assertEqual(xxdd[0].difforder, 2)

        sigma1 = (6, 7, 8)

        c_star = a1*a3*a6*x1*7

        # consistency check for local calculations
        self.assertEqual(Mu_1.get_coeff_from_idcs(sigma1), c_star)

        # in the following dos means difforder symbol
        res1, dos = pc.coeff_ido_derivorder(sigma1, mu1, mu2, mu3, tds=aa)
        self.assertEqual(Mu_1.get_coeff_from_idcs(sigma1), res1)

        # test array type for sigma
        sigma_arr = pc.np.array([6.0, 7.0, 8.0])
        res1b, dos = pc.coeff_ido_derivorder( sigma_arr, mu1, mu2, mu3, tds=aa)
        self.assertEqual(Mu_1.get_coeff_from_idcs(sigma1), res1b)

        sigma2 = (5, 7, 8)
        res2, dos = pc.coeff_ido_derivorder(sigma2, mu1, mu2, mu3, tds=aa)
        self.assertEqual(res2, 0)

        sigma3 = (5, 6, 7)
        res3, dos = pc.coeff_ido_derivorder(sigma3, mu1, mu2, mu3, tds=aa)
        assert res3.has(dos)
        self.assertEqual(Mu_1.get_coeff_from_idcs(sigma3), res3.subs(dos, 1))

        mu1.jet_extend_basis()
        mu2.jet_extend_basis()
        mu3.jet_extend_basis()

        Mu_2 = mu1.dot(aa).dot(aa)^mu2.dot(aa).dot(aa)^mu3.dot(aa).dot(aa)
        sigma4 = (5+3, 6+3, 7+3)
        res4 = Mu_2.get_coeff_from_idcs(sigma4)
        self.assertEqual(res4, res3.subs(dos, 2))

        with self.assertRaises(ValueError) as cm:
            sigma5 = (1, 6, 7)
            pc.coeff_ido_derivorder(sigma5, mu1, mu2, mu3, tds=aa)

    def test_coeff_ido_do2(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xxd = pc.st.time_deriv(xx, xx)

        full_basis = pc.st.row_stack(xx, xxd)
        foo, ddx = pc.setup_objects(full_basis)
        dx1, dx2, dx3, dxdot1, dxdot2, dxdot3 = ddx

        aa = sp.Matrix(sp.symbols("a1:10"))
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = aa

        mu1 = a1*dxdot1 + 3*a2*dx2
        mu2 = a3*dxdot2 - a4*dx1 + 6*a5*dx2
        mu3 = 7*a6*x1*dxdot3 + a7*dx3

        sigma1 = (6, 7, 8)
        sigma1b = (6+3, 7+3, 8+3)

        sigma3 = (5, 6, 7)

        with self.assertRaises(ValueError) as cm:
            # the indices are too high
            pc.coeff_ido_derivorder(sigma1b, mu1, mu2, mu3)

        # important to call this before mu_i.jet_extend_basis()
        res2, dos = pc.coeff_ido_derivorder(sigma1, mu1, mu2, mu3, tds=aa)
        res3, dos = pc.coeff_ido_derivorder(sigma3, mu1, mu2, mu3, tds=aa)

        mu1.jet_extend_basis()
        mu2.jet_extend_basis()
        mu3.jet_extend_basis()

        Mu_0 = mu1^mu2^mu3
        Mu_1 = mu1.dot(aa)^mu2.dot(aa)^mu3.dot(aa)

        res1, dos = pc.coeff_ido_derivorder(sigma1, mu1, mu2, mu3)
        self.assertEqual(Mu_1.get_coeff_from_idcs(sigma1), res2)
        self.assertEqual(Mu_1.get_coeff_from_idcs(sigma1), res1)

        self.assertEqual(Mu_1.get_coeff_from_idcs(sigma3), res3.subs(dos, 1))

    # noinspection PyPep8
    def test_hodge_star_3d(self):

        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        foo, ddx = pc.setup_objects(xx)
        dx1, dx2, dx3 = ddx

        # cross product is hodge dual of wedge product
        self.assertEqual((dx1^dx2).hodge_star(), dx3)
        self.assertEqual((dx1^dx3).hodge_star(), -dx2)
        self.assertEqual((dx2^dx3).hodge_star(), dx1)

        # https://en.wikipedia.org/wiki/Hodge_isomorphism#Three_dimensions
        self.assertEqual(dx1.hodge_star(), dx2^dx3)
        self.assertEqual(dx2.hodge_star(), dx3^dx1)  # note the non canonical order
        self.assertEqual(dx3.hodge_star(), dx1^dx2)

    @unittest.expectedFailure
    def test_hodge_star_4d(self):
        # noinspection PyPep8Naming

        x1, x2, x3, x4 = xx = sp.Matrix(sp.symbols("x1, x2, x3, x4"))
        foo, ddx = pc.setup_objects(xx)
        dt, dx, dy, dz = ddx

        # see https://en.wikipedia.org/wiki/Hodge_isomorphism#Four_dimensions
        # Minkowski spacetime with metric signature (+ − − −)
        # Currently not supported

        self.assertEqual(dt.hodge_star(), dx^dy^dz)
        self.assertEqual(dx.hodge_star(), dt^dy^dz)
        self.assertEqual(dy.hodge_star(), -dt^dx^dz)
        self.assertEqual(dz.hodge_star(), dt^dx^dy)

        self.assertEqual((dt^dx).hodge_star(), -dy^dz)


# noinspection PyUnresolvedReferences
# noinspection PyPep8Naming
class TestVectorDifferentialForms(unittest.TestCase):
    def setUp(self):
        pass

    def test_vector_k_form(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4')
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))

        w1 = pc.DifferentialForm(1, XX, coeff=Q_[0,:])
        w2 = pc.DifferentialForm(1, XX, coeff=Q_[1,:])

        w = pc.VectorDifferentialForm(1, XX, coeff=Q_)
        w1_tilde = w.get_differential_form(0)
        self.assertEqual(w1.coeff, w1_tilde.coeff)

        w2_tilde = w.get_differential_form(1)
        self.assertEqual(w2.coeff, w2_tilde.coeff)

    def test_vector_form_get_coeff_from_idcs(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4')
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))
        w = pc.VectorDifferentialForm(1, XX, coeff=Q_)
        w_0 = w.get_coeff_from_idcs(0)
        Q_0 = Q_.col(0)
        self.assertEqual(w_0, Q_0)

        w_1 = w.get_coeff_from_idcs(1)
        Q_1 = Q_.col(1)
        self.assertEqual(w_1, Q_1)

    def test_unpack_vector_form(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        s  = sp.Symbol('s', commutative=False)
        C  = sp.Symbol('C', commutative=False)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))

        # 1-forms
        w1 = pc.DifferentialForm(1, XX, coeff=Q_[0,:])
        w2 = pc.DifferentialForm(1, XX, coeff=Q_[1,:])

        # vector 1-form
        w = pc.VectorDifferentialForm(1, XX, coeff=Q_)

        w1_unpacked, w2_unpacked = w.unpack()

        self.assertEqual(w1.coeff, w1_unpacked.coeff)
        self.assertEqual(w2.coeff, w2_unpacked.coeff)

    def test_vector_form_append(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        s  = sp.Symbol('s', commutative=False)
        C  = sp.Symbol('C', commutative=False)

        # vector 1-form
        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])

        Q_ = st.col_stack(Q, sp.zeros(2, 6))
        w = pc.VectorDifferentialForm(1, XX, coeff=Q_)

        # 1-forms
        Q2 = sp.Matrix([
            [x1, x2, x3]])

        Q2_ = st.col_stack(Q2, sp.zeros(1, 6))
        w2 = pc.DifferentialForm(1, XX, coeff=Q2_[0,:])

        w.append(w2)

        # vector form to compare with:
        B = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3],
            [x1, x2, x3]])

        B_ = st.col_stack(B, sp.zeros(3, 6))
        b = pc.VectorDifferentialForm(1, XX, coeff=B_)

        self.assertEqual(w.coeff, b.coeff)

    def test_vector_form_append_2(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        s  = sp.Symbol('s', commutative=False)
        C  = sp.Symbol('C', commutative=False)

        # vector 1-form
        Q1 = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])

        Q1_ = st.col_stack(Q1, sp.zeros(2, 6))
        w1 = pc.VectorDifferentialForm(1, XX, coeff=Q1_)

        # 1-forms
        Q2 = sp.Matrix([
            [x1, x2, x3],
            [x3, x1, x2]])

        Q2_ = st.col_stack(Q2, sp.zeros(2, 6))
        w2 = pc.VectorDifferentialForm(1, XX, coeff=Q2_)

        w1.append(w2)

        # vector form to compare with:
        B = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3],
            [x1, x2, x3],
            [x3, x1, x2]])
        B_ = st.col_stack(B, sp.zeros(4, 6))

        self.assertEqual(w1.coeff, B_)

    def test_stack_to_vector_form(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))

        # 1-forms
        w1 = pc.DifferentialForm(1, XX, coeff=Q_[0,:])
        w2 = pc.DifferentialForm(1, XX, coeff=Q_[1,:])
        w_stacked = pc.stack_to_vector_form(w1, w2)

        # vector 1-form
        w = pc.VectorDifferentialForm(1, XX, coeff=Q_)

        self.assertEqual(w.coeff, w_stacked.coeff)

    def test_mul(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)

        s = sp.Symbol('s', commutative=False)
        C = sp.Symbol('C', commutative=False)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])

        W = pc.VectorDifferentialForm(1, xx, coeff=Q)

        W1 = s*W
        W2 = W*C

        self.assertEqual(W1.coeff, nct.nc_mul(s,W.coeff))
        self.assertNotEqual(W1.coeff, nct.nc_mul(W.coeff,s))

        self.assertEqual(W2.coeff, nct.nc_mul(W.coeff,C))
        self.assertNotEqual(W2.coeff, nct.nc_mul(C,W.coeff))

        alpha = pc.DifferentialForm(1, xx)
        with self.assertRaises(TypeError) as cm:
            alpha*W1
        with self.assertRaises(TypeError) as cm:
            W1*alpha

        M = sp.eye(2)
        with self.assertRaises(TypeError) as cm:
            M*W1
        with self.assertRaises(TypeError) as cm:
            W1*M

    def test_left_mul_by_1(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        s  = sp.Symbol('s', commutative=False)
        C  = sp.Symbol('C', commutative=False)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))

        # s-dependent matrix
        M1 = sp.Matrix([
            [1,0],
            [-C*s,1]])

        # 1-forms
        w1 = pc.DifferentialForm(1, XX, coeff=Q_[0,:])
        w2 = pc.DifferentialForm(1, XX, coeff=Q_[1,:])

        # vector 1-form
        w = pc.VectorDifferentialForm(1, XX, coeff=Q_)

        t = w.left_mul_by(M1, s, [C])
        t2 = -C*w1.dot() + w2

        self.assertEqual(t2.coeff, t.coeff.row(1).T)

    def test_left_mul_by_2(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        C  = sp.Symbol('C', commutative=False)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))

        # matrix independent of s
        M2 = sp.Matrix([
            [1,0],
            [-C,1]])

        # 1-forms
        w1 = pc.DifferentialForm(1, XX, coeff=Q_[0,:])
        w2 = pc.DifferentialForm(1, XX, coeff=Q_[1,:])

        # vector 1-form
        w = pc.VectorDifferentialForm(1, XX, coeff=Q_)

        t = w.left_mul_by(M2, additional_symbols=[C])
        # object to compare with:
        t2 = -C*w1 + w2

        self.assertEqual(t2.coeff, t.coeff.row(1).T)

    def test_left_mul_by_3(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        s  = sp.Symbol('s', commutative=False)
        C  = sp.Symbol('C', commutative=False)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))

        M3 = sp.Matrix([
            [1,0],
            [-C*s**2,1]])

        # vector 1-forms
        w = pc.VectorDifferentialForm(1, XX, coeff=Q_)

        with self.assertRaises(Exception) as cm:
            # raises NotImplemented but this might change
            t = w.left_mul_by(M3, s, additional_symbols=[C])

    def test_vector_form_subs(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        C  = sp.Symbol('C', commutative=False)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [C, 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))

        B = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [-tan(x1), 0, x3]])
        B_ = st.col_stack(B, sp.zeros(2, 6))

        # vector 1-forms
        omega = pc.VectorDifferentialForm(1, XX, coeff=Q_).subs(C, -tan(x1))

        self.assertEqual(B_, omega.coeff)

    def test_sum_two_vector_forms(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        A = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [1, 0, x3]])
        A_ = st.col_stack(A, sp.zeros(2, 6))

        B = sp.Matrix([
            [x3/cos(x1), 0, 1],
            [-tan(x1), 0, x3]])
        B_ = st.col_stack(B, sp.zeros(2, 6))

        # vector 1-forms
        omega_a = pc.VectorDifferentialForm(1, XX, coeff=A_)
        omega_b = pc.VectorDifferentialForm(1, XX, coeff=B_)

        omega_c = omega_a + omega_b

        # vector form to compare with
        Q_ = A_ + B_
        omega_comp = pc.VectorDifferentialForm(1, XX, coeff=Q_)

        self.assertEqual(omega_c.coeff, omega_comp.coeff)

    def test_difference_vector_forms(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        A = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [1, 0, x3]])
        A_ = st.col_stack(A, sp.zeros(2, 6))

        B = sp.Matrix([
            [x3/cos(x1), 0, 1],
            [-tan(x1), 0, x3]])
        B_ = st.col_stack(B, sp.zeros(2, 6))

        # vector 1-forms
        omega_a = pc.VectorDifferentialForm(1, XX, coeff=A_)
        omega_b = pc.VectorDifferentialForm(1, XX, coeff=B_)

        omega_c = omega_a - omega_b

        # vector form to compare with
        Q_ = A_ - B_
        omega_comp = pc.VectorDifferentialForm(1, XX, coeff=Q_)

        self.assertEqual(omega_c.coeff, omega_comp.coeff)

    def test_vector_form_dot(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [1, 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))

        # vector 1-forms
        omega = pc.VectorDifferentialForm(1, XX, coeff=Q_)
        omega_dot = omega.dot()


        omega_1, omega_2 = omega.unpack()

        omega_1dot = omega_1.dot()
        omega_2dot = omega_2.dot()
        Qdot_ = st.row_stack(omega_1dot.coeff.T, omega_2dot.coeff.T)
        # vector form to compare with
        omegadot_comp = pc.VectorDifferentialForm(1, XX, coeff=Qdot_)

        self.assertEqual(omega_dot.coeff, omegadot_comp.coeff)

    def test_vector_form_dot(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4', commutative=False)
        xdot1, xdot2, xdot3 = xxdot = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxddot = st.time_deriv(xxdot, xxdot)

        XX = st.row_stack(xx, xxdot, xxddot)

        Q = sp.Matrix([
            [x3/sin(x1), 1, 0],
            [1, 0, x3]])
        Q_ = st.col_stack(Q, sp.zeros(2, 6))

        # vector 1-forms
        omega = pc.VectorDifferentialForm(1, XX, coeff=Q_)

        with self.assertRaises(ValueError) as cm:
            # coordinates are not part of basis
            omega.dot().dot().dot()

    def test_simplify(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4',)

        Q = sp.Matrix([[sin(x1)**2 + cos(x1)**2 - 1, 0, 0],
                       [0, x3*(1 - sin(x2)**2) -cos(x2)**2*x3, 0]])

        Omega = pc.VectorDifferentialForm(1, xx, coeff=Q)
        Omega2 = pc.simplify(Omega)
        self.assertEqual(Omega2.coeff, 0*Omega.coeff)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
