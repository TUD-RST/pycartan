# -*- coding: utf-8 -*-
"""
Created on Tue Sep 03 17:44:46 2013

@author: Torsten Knüppel (original Code)
@author: Carsten Knoll (enhancements)
@author: Klemens Fritzsche (enhancements)
"""

from six import string_types  # py2 and 3 compatibility

from itertools import combinations
import numpy as np
import sympy as sp
from sympy.core.sympify import CantSympify
import itertools as it
import inspect

import symbtools as st  # needed for make_global, time_deriv, srn
import symbtools.noncommutativetools as nct
from symbtools import lzip
from functools import reduce

from ipydex import IPS  # for debugging

# TODO: compare to
# from sympy.functions.special.tensor_functions import eval_levicivita
def sign_perm(perm):
    """
    :param perm:
    :return: the sign of a permutation (-1 or 1)
    examples:  [0, 1, 2] -> 1;
               [1, 0, 2]) -> -1
               [1, 0, 3]) -> -1
    """
    perm = range_indices(perm)  # prevent problems with non consecutive indices
    Np = len(np.array(perm))
    sgn = 1
    for n in range(Np - 1):
        for m in range(n + 1, Np):
            sgn *= 1. * (perm[m] - perm[n]) / (m - n)
    sgn = int(sgn)

    assert sgn in (-1, 1), "error %s" % perm

    return sgn


# TODO: duplicate of sign_perm
def perm_parity(seq):
    """
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    """
    # adapted from http://code.activestate.com/
    # recipes/578227-generate-the-parity-or-sign-of-a-permutation/
    # by Paddy McCarthy
    lst = range_indices(seq)  # normalize the sequence to the first N integers
    parity = 1
    index_list = list(range(len(lst)))
    for i in range(0, len(lst) - 1):
        if lst[i] != i:
            # there must be a smaller number in the remaining list
            # -> perform an exchange
            #            print i, lst
            mn = np.argmin(lst[i:]) + i
            lst[i], lst[mn] = lst[mn], lst[i]
            #            print mn, lst
            parity *= -1
    return parity


def range_indices(seq):
    """
    returns a tuple which represents the same permutation
    but with indices from range(N)
    (5, 2, 10, 1) -> (2, 1, 3, 0)
    """
    assert len(set(seq)) == len(seq)

    new_elements = list(range(len(seq)))

    res = [None] * len(seq)

    seq = list(seq)
    seq_work = list(seq)

    for i in range(len(seq)):
        m1 = min(seq_work)
        m2 = min(new_elements)

        i = seq.index(m1)
        res[i] = m2

        seq_work.remove(m1)
        new_elements.remove(m2)

    assert not None in res
    return res


class DifferentialForm(CantSympify):
    def __init__(self, n, basis, coeff=None, name=None):
        """
        :n: degree (e.g. 0-form, 1-form, 2-form, ... )
        :basis: list of basis coordinates (Symbols)
        :coeff: coefficient vector for initilization (defualt: [0, ..., 0])
        :name: optional Name

        """

        self.grad = n
        self.basis = sp.Matrix(basis)
        self.dim_basis = len(basis)
        # list of allowed indices
        if (self.grad == 0):
            self.indizes = [(0,)]
        else:
            # this is a list like [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            #TODO: this should be renamed to index_tuples
            self.indizes = list(combinations(list(range(self.dim_basis)), self.grad))

        # number of coefficient
        self.num_coeff = len(self.indizes)

        # coefficients of the differential form
        if coeff is None:
            self.coeff = sp.zeros(self.num_coeff, 1)
        else:
            assert len(coeff) == self.num_coeff
            # TODO: use a row vector here
            self.coeff = sp.Matrix(coeff).reshape(self.num_coeff, 1)

        self.name = name  # useful for symbtools.make_global

    # quick hack combine new name with backward compability
    @property
    def koeff(self):
        return self.coeff

    @property
    def c(self):
        return self.coeff

    @property
    def degree(self):
        return self.grad

    @property
    def indices(self):
        return self.indizes

    def __repr__(self):
        #return self.ausgabe()
        return self.to_str()

    def __add__(self, a):
        assert self.grad == a.grad
        assert self.basis == a.basis

        new_form = DifferentialForm(self.grad, self.basis)
        new_form.coeff = self.coeff + a.coeff

        return new_form

    def __sub__(self, m):

        return self + (m * (-1))

    def __rmul__(self, f):
        """multiplication from the left (reverse mul) -> f*self
        This method provides wedge-product and scalar multiplication

        see also docstring of `__mul__`
        """

        if isinstance(f, DifferentialForm):
            # normally this should not happen because `f.__mul__` is executed
            # anyway, no problem:
            return f.wp(self)

        # scalar multiplication -> use commutativity
        return self * f

    def __mul__(self, f):
        """ multiplication from the right
        This method provides wedge-product and scalar multiplication

        Note: for Differential Forms w1, w2, w3 we have:
        w1*w2 == w1^w2
        but due to python operator precedence of `+` over `^` we have
        w1*w2+w3 == (w1*w2)+w3 != w1^w2+w3 == w1^(w2+w3)
        """

        if isinstance(f, DifferentialForm):
            return self.wp(f)

        if st.is_scalar(f):
            # now scalar multiplication
            new_form = DifferentialForm(self.grad, self.basis)

            new_form.coeff = self.coeff * f
            return new_form
        else:
            msg = "Multiplication not implemented for types: %s and %s" % ( type(self), type(f) )
            raise TypeError(msg)

    def __div__(self, arg):
        # python2 leagacy compatibility
        return self.__truediv__(arg)

    def __truediv__(self, arg):
        msg = "Unexpected arg for div: %s of type(%s)" % (str(arg), type(arg))
        try:
            arg = sp.sympify(arg)
        except sp.SympifyError:
            raise TypeError(msg)
        if not isinstance(arg, sp.Basic):
            raise TypeError(msg)

        return (1/arg)*self


    def __xor__(self, f):
        """
        overload ^-operator with wedge product

        caveat: possible pitfall because of pythons operator precedence:
            dx1^dx2 + dx3^dx4 ist evaluated like this:
            dx1^(dx2 + dx3)^dx4 which is not intuitive
        """
        return self.wp(f)

    def __eq__(self, other):
        """test for equality"""
        assert isinstance(other, DifferentialForm)
        assert self.basis == other.basis
        if self.grad == other.grad:
            return self.coeff == other.coeff
        else:
            return False

    def __neg__(self):
        return -1 * self

    # TODO: where is this needed? Could it implemented as explicit call?
    def __getitem__(self, ind):
        """return the coefficient, corresponding to the index-tuple ind"""
        ind = np.atleast_1d(ind)
        assert len(ind) == self.grad
        try:
            ind_1d, vz = self.__getindexperm__(ind)
        except ValueError as ve:
            raise ValueError("Invalid Index; " + ve.message)
        else:
            return vz * self.coeff[ind_1d]

    # TODO: where is this needed? Could it implemented as explicit call?
    def __setitem__(self, ind, wert):
        self.setitem(ind, wert)

    def setitem(self, idx_tup, value):
        """set the coefficient corresponding to the index-tuple ind"""
        idx_tup = np.atleast_1d(idx_tup)
        assert len(idx_tup) == self.grad
        try:
            idx_1d, sign = self.__getindexperm__(idx_tup)
        except ValueError:
            errmsg = 'invalid index-tuple: %s' % str(idx_tup)
            raise ValueError(errmsg)
        else:
            self.coeff[idx_1d] = sign * value

    # TODO: Should be reformulated as _get_canonical_signed_index
    def __getindexperm__(self, ind):
        """ Liefert den 1d-Index und das Vorzeichen der Permutation"""
        if len(ind) == 1:
            ind_1d = self.indizes.index(tuple(ind))
            sgn = 1
        else:
            srt_arr = np.argsort(ind)
            #vz = sign_perm(srt_arr)
            sgn = perm_parity(srt_arr)
            ind_1d = self.indizes.index(tuple(ind[srt_arr]))
        return ind_1d, sgn

    def diff(self):
        """returns the exterior derivative"""
        # create new form (with 1 degree higher) for the result
        res = DifferentialForm(self.grad + 1, self.basis)
        # 0-Form separately
        if self.grad == 0:
            for m in range(self.dim_basis):
                res[m] = sp.diff(self.coeff[0], self.basis[m])
        else:
            for n in range(self.num_coeff):
                # get the index-tuple corresponding to coeff-entry n
                ind_n = self.indizes[n]

                for m in range(self.dim_basis):
                    if m in ind_n:
                        # m is already contained in the n-th index-tuple
                        # this is the 'dx1^dx1' case.
                        continue
                    else:
                        new_idx_tpl = (m,) + ind_n
                        # calculate coeff for dx_m ^ ...
                        # this result contributes to the coeff of the
                        # canocnical basis form dx_1^...^dx_m^...^dx_N
                        # the sign is respected by __getitem__ and __setitem__
                        partial_coeff = sp.diff(self.coeff[n], self.basis[m])
                        res[new_idx_tpl] += partial_coeff

        return res

    @property
    def d(self):
        """property -> allows for short syntax: w1.d (= w1.diff())"""
        return self.diff()

    def nonzero_tuples(self, srn=False, eps=1e-30):
        """
        returns a list of tuples (coeff, idcs) for each index-tuple idcs,
        where the corresponding coeff != 0
        """
        Z = lzip(self.coeff, self.indices)

        if srn == "prime":
            res = [(c, idcs) for (c, idcs) in Z if abs(st.subs_random_numbers(c, prime=True)) > eps]
        elif srn:
            res = [(c, idcs) for (c, idcs) in Z if abs(st.subs_random_numbers(c)) > eps]
        else:
            res = [(c, idcs) for (c, idcs) in Z if c != 0]

        return res

    def is_zero(self, n=100):
        """
        simplifies zero candidates with count_ops <  n
        """

        Z = self.nonzero_tuples()
        for c, idcs in Z:
            if sp.count_ops(c) < n:
                c = sp.simplify(c)
            if c != 0:
                return False
        return True

    def simplify(self, *args, **kwargs):
        """
        calls the simplify method of the coeff-matrix
        (nothing is returned)
        """
        self.coeff.simplify(*args, **kwargs)

    def expand(self, *args, **kwargs):
        """
        returns a copy of this form with expand() applied to its coeff-matrix
        """
        res = DifferentialForm(self.grad, self.basis)
        res.coeff = self.coeff.expand(*args, **kwargs)
        return res

    def subs(self, *args, **kwargs):
        """
        returns a copy of this form with subs(...) applied to its coeff-matrix
        """
        res = DifferentialForm(self.grad, self.basis)
        res.coeff = self.coeff.subs(*args, **kwargs)
        return res

    def change_of_basis(self, new_basis, Psi, Psi_jac=None):
        """
        let x1, ... be coordinates of the actual (old) basis
        and y1, ... of the new basis.
        y = Phi(x), x = Psi(y)

        a 1-form is given by w = sum( ci(x)*dxi ) = row(c(x))*col(dx)
        dx:= Psi_jac(y)*dy

        w_new = row( c(Psi(y))*Psi_jac(y)*dy )

        c(Psi(y))*Psi_jac(y) is the new coeff-row-vector
        (allthough stored as column)

        Psi_jac might by supplied for performance reasons
        """
        assert len(new_basis) == len(self.basis)
        assert self.grad <= 1  # not yet understood for higher order
        res = DifferentialForm(self.grad, new_basis)
        k = self.coeff.subs(zip(self.basis, Psi))

        if Psi_jac == None:
            Psi_jac = Psi.jacobian(new_basis)

        k_new = (k.T * Psi_jac).T
        assert k_new.shape == (len(new_basis), 1)

        res.coeff = k_new
        return res

    @property
    def ord(self):
        """returns the highest differential order of the highest nonzero coefficient

        example: w=(dx1^dxdot1) + (dx2^dxdddot1)
           w.ord -> 3
        """
        base_length = self._calc_base_length()
        nzt = self.nonzero_tuples(srn=True)

        if len(nzt) == 0:
            return 0

        # get the highest scalar index (basis index) which occurs in any nonzero index tuple

        nz_idx_array = np.array(lzip(*nzt)[1])
        highest_base_index = np.max(nz_idx_array)

        res = int(highest_base_index/base_length)

        return res


    def get_coeff(self, base_form):
        """
        if self == 7*dx - x**2*dy,
        self.get_coeff(dx) -> 7
        self.get_coeff(-x*dy) -> x
        """

        assert self.basis == base_form.basis
        assert self.grad == base_form.grad

        nzi = base_form.nonzero_tuples()
        assert len(nzi) == 1

        coeff, idcs = nzi[0]
        res = self[idcs] / coeff

        return res

    def get_coeff_from_idcs(self, idcs):
        """
        Abstraction wrapper for self.__getitem__

        :param idcs:
        :return:
        """

        return self[idcs]

    def get_baseform_from_idcs(self, idcs):
        """
        Expects an N-tuple of integers and returns the baseform of self corresponding
        to that index tuple.

        :param idcs:
        :return:
        """

        if not isinstance(idcs, (int, list, tuple, sp.Matrix, st.np.ndarray)):
            msg = "Expected tuple of integers, got: %s" % type(idcs)
            raise TypeError(msg)

        if isinstance(idcs, int):
            idcs = (idcs, )

        idcs = tuple(idcs)

        if not len(idcs) == self.degree:
            msg = "Expected index tuple of lenght %i but got length %i"
            msg = msg % (self.degree, len(idcs))
            raise ValueError(msg)

        if not idcs in self. indices:
            msg = "Got unexpexted index tuple: %s" % str(idcs)
            raise ValueError(msg)

        res = DifferentialForm(self.degree, self.basis)
        res[idcs] = 1

        return res

    def get_baseform_from_plain_index(self, idx):
        """
        returns the i-th baseform (corresponding to the i-th element of
        self.coeff)
        """
        assert int(idx) == idx
        N = self.num_coeff
        if not -N <= idx < N:
            raise ValueError("%i not in range [0, ..., %i]" %(idx, N-1) )

        idx_tup = self.indices[idx]
        return self.get_baseform_from_idcs(idx_tup)

    def get_multiplied_baseform(self, arg):
        """
        Returns the baseform multiplied with the appropriate element of `self.coeff`,
        corresponding to `arg`

        arg:    either a basis-component or an appropriate index-tuple

        Let self == 7*dx^dy + 3*y*dx^dz (2-form over basis x, y, z)
        self.get_component(dx^dz) -> 3*y*dx^dz
        """

        if isinstance(arg, (list, tuple, st.np.ndarray)):
            arg = tuple(arg)
            if not arg in self.indices:
                msg = "Invalid index-tuple: " + str(arg)
                raise ValueError(msg)
            res_idcs = arg

        else:
            assert isinstance(arg, DifferentialForm)
            if not arg.degree == self.degree:
                msg = "Wrong degree (%i instead of %i)" % (arg.degree, self.degree)
                raise ValueError(msg)

            # arg should be decomposable -> only one nonzero tuple
            nzt = arg.nonzero_tuples()

            if not len(nzt) == 1:
                msg = "Unexpeced (not decomposable) form supplied. Only base-forms are supported"
                raise ValueError(msg)

            # additionally: coeff should be 1
            if not nzt[0][0] == 1:
                msg = "Unexpeced form supplied. Only base-forms are supported. Coeff != 1"
                raise ValueError(msg)

            res_idcs = nzt[0][1]

        res = self*0
        res[res_idcs] = self[res_idcs]

        return res

    # TODO: Refactoring
    # Differentialform ausgeben

    def _idcs_to_str(self, idcs, latex=None):
        """convert an index tuple like (0, 2, 3) into an string like "dx1^dx3^dx4"
        """

        if not latex:
            coord_strings = ["d"+ (self.basis[i]).name for i in idcs]
            product_string = "^".join(coord_strings)
        else:
            coord_strings = [r"\d "+ (latex(self.basis[i])) for i in idcs]
            product_string = r"\wedge".join(coord_strings)
        return product_string


    def to_str(self):
        if self.grad == 0:
            return str(self.coeff[0])
        nztuples = [(idcs, coeff) for idcs, coeff in zip(self.indices, self.coeff) if coeff != 0]

        if self.grad > self.dim_basis:
            coord_strings = ['d'+self.basis[0].name] * self.grad
            return "(0)" + "^".join(coord_strings)

        res_strings = []
        for idcs, coeff in nztuples:
            tmp_str = "(%s)%s" %(str(coeff), self._idcs_to_str(idcs) )
            res_strings.append(tmp_str)

        if len(res_strings) == 0:
            res = "(0)%s" % self._idcs_to_str(self.indices[0])
        else:
            res = "  +  ".join(res_strings)
        return res

    def to_latex(self, latex_func=sp.latex):
        if self.grad == 0:
            return str(self.coeff[0])
        nztuples = [(idcs, coeff) for idcs, coeff in zip(self.indices, self.coeff) if coeff != 0]

        res_strings = []
        for idcs, coeff in nztuples:
            tmp_str = "%s %s" %(latex_func(coeff), self._idcs_to_str(idcs, latex=latex_func) )
            res_strings.append(tmp_str)

        if len(res_strings) == 0:
            res = "(0)%s" % self._idcs_to_str(self.indices[0])
        else:
            res = "  +  ".join(res_strings)
        return res

    def ausgabe(self):
        # 0-Form separat
        if self.grad == 0:
            df_str = str(self.coeff[0])
        else:
            df_str = '0'
            for n in range(self.num_coeff):
                coeff_n = self.coeff[n]
                ind_n = self.indizes[n]
                if coeff_n == 0:
                    continue
                else:
                    # String of the coefficient
                    sub_str = '(' + self.eliminiere_Ableitungen(
                        self.coeff[n]) + ') '
                    # Füge Basis-Vektoren hinzu
                    for m in range(self.grad - 1):
                        sub_str += 'd' + (self.basis[ind_n[m]]).name + '^'
                    sub_str += 'd' + (self.basis[ind_n[self.grad - 1]]).name
                # Gesamtstring
                if df_str == '0':
                    df_str = sub_str
                else:
                    df_str += '+' + sub_str
        # Ausgabe
        return df_str

    # TODO: Refactoring
    def eliminiere_Ableitungen(self, coeff):
        coeff = sp.simplify(coeff)
        at_deri = list(coeff.atoms(sp.Derivative))
        if at_deri:
            for deri_n in at_deri:
                # Argumente des Derivative-Objekts
                deri_args = deri_n.args
                # Name der differenzierten Funktion
                name_diff = str(type(deri_args[0]))
                # Argumente der differenzierten Funktion
                diff_arg = deri_args[0].args
                # Ableitungsordnungen ermitteln
                ndiff = list()
                for m in diff_arg:
                    ndiff.append(deri_args[1:].count(m))
                # Neue Funktion mit entsprechenden Argumenten erzeugen
                dfunc = sp.Function('D' + str(ndiff) + name_diff)(*diff_arg)
                # Neue Funktion einsetzen und Substitutionen durchführen
                coeff = coeff.subs(deri_n, dfunc).doit()
        return str(coeff)

    def hodge_star(self):
        """
        returns the hodge dual of this form

        assumptions: scalar product is given by identity matrix (implies signature 0)

        Background: see Chapter 1, example 2 in Agricola, Friedrich: Global Analysis -
        Differential Forms in Analysis, Geometry and Physics.
        """

        # for every coeff we need its index-tuple I and the complementrary index-tuple J
        # and then the signature

        all_indices = set(range(self.dim_basis))
        result = DifferentialForm(self.dim_basis - self.grad, self.basis)
        for counter, I in enumerate(self.indices):
            # complementary tuple
            J = sorted(all_indices - set(I))
            permutation = list(I)
            permutation.extend(J)

            s = sign_perm(permutation)
            result[J] = s*self[I]

        return result

    # TODO: extend to higher degrees (maybe)
    def integrate(self):
        """
        assumes self to be a 1-Form
        if self.d.is_zero() then this method returns h such that:
            d h = self
        """

        if not self.grad == 1:
            raise NotImplementedError("not yet supported")

        if not self.d.is_zero():
            msg = "This form seem not to be closed (self.d != 0 )"
            raise ValueError(msg)

        complete_basis = list(self.basis)

        r = st.sca_integrate(self.coeff[0], self.basis[0]).simplify()

        total_deriv = d(r, self.basis)

        difference = simplify(self - total_deriv)

        if not difference.coeff[0] == 0:
            # this is my assumption
            msg = "Unexpected result while integration of 1-form"
            raise ValueError(msg)

        if difference.is_zero():
            result = r
        else:
            # the actual result is r + IC(x2, ..., xn)
            # now determine IC from the equation: d(IC) = difference

            if self.dim_basis == 1:
                # sympy.integrate should have solved this problem
                msg = "Unexpected result while integration of 1-form"
                raise ValueError(msg)

            assert self.dim_basis > 1

            # integration constant might depend on all remaning vars
            reduced_basis = complete_basis[1:]
            delta = DifferentialForm(1, reduced_basis, difference.coeff[1:])

            IC = delta.integrate()
            result = sp.simplify(r + IC)

        # final test: take the exterior derivative and compare to self
        result_d_coeff = sp.simplify( d(result, self.basis).coeff )
        self_coeff = sp.simplify(self.coeff)
        if not result_d_coeff == self_coeff:
            msg = "Unexpected final result while calculating integration constants"
            # IPS()
            raise ValueError(msg)

        return result

    # TODO: merge with the contract function
    def contract(self, vf):
        """
        Contract this differential form with a vectorfield
        (appply the form on it)

        We assume the same basis

        """

        if not self.grad == 1: raise NotImplementedError("not yet supported")
        assert vf.shape == self.coeff.shape
        res = self.coeff.T * vf
        return res[0, 0]

    def wp(self, *args):
        """
        convenience method
        return wp(self, args) (wedgeproduct)
        """
        return wp(self, *args)

    def rank(self):
        """
        computes the generic rank of the differential form (assume degree == 1)
        this relies on simplification because of a comparision with zero
        """

        assert self.grad == 1, "Definition only valid for one-forms"

        s = self
        ds = self.diff()

        r = 0
        test_form = wp(ds, s)  # contains (ds)**(r+1) ^ s
        while True:
            test_form.coeff.simplify()

            if test_form.coeff == test_form.coeff * 0:
                return r

            r += 1
            test_form = wp(ds, test_form)

            assert r <= (self.dim_basis - 1) / 2.0

    def jet_extend_basis(self, order=1, zero_order_hint=None):
        """
        suppose self = a*dx1 + b*dx2
        after application of this method we have
        self = a*dx1 + b*dx2 + 0*dxdot1 + 0*dxdot2

        This is needed in the context of differential forms,
        defined on jet bundles, where d/dt (self) the coefficients might
        depend on (time-) derivatives of the basis coordinates, e.g a = 3*xdot1.
        To properly calculate self.d (the extrerior derivative) the basis must be
        prepared for components 'in direction of' dxdot1
        """
        old_basis = sp.Matrix(self.basis)
        old_coeff = self.coeff
        old_indices = self.indices  # List of k-tuples (self.degree == k)

        assert old_basis.shape[1] == 1

        if order != 1:
            raise NotImplementedError("Not yet done.")

        # we need to determine which are the symbols, corresponding to
        # the 0th order coordinates
        # in the future this will be done by checking a special attribute
        # of the ExtendedSymbols.
        # Currently, we use a manual hint

        if zero_order_hint:
            zoh = sp.Matrix(zero_order_hint)
        else:
            # assume, we only have 0th order
            zoh = sp.Matrix([elt for elt in self.basis if elt.difforder == 0])

        L = len(zoh)
        assert zoh.shape[1] == 1
        assert old_basis[:L, :] == zoh
        N = len(old_basis) * 1.0/L
        assert int(N) == N

        # old_highest_derivs
        ohd = old_basis[-L:, :]

        new_highest_derivs = st.time_deriv(ohd, old_basis, order=1)

        new_basis = st.row_stack(old_basis, new_highest_derivs)

        new_form  = DifferentialForm(self.degree, new_basis)
        self.basis = new_basis
        self.dim_basis = len(new_basis)
        self.indizes = new_form.indices
        self.coeff = new_form.coeff
        self.num_coeff = new_form.num_coeff

        for idx_tup in self.indices:
            if idx_tup in old_indices:
                # get the index of the index-tuple
                meta_idx =  old_indices.index(idx_tup)
                c = old_coeff[meta_idx]
            else:
                c = 0
            self.setitem(idx_tup, c)

    def _calc_base_length(self):

        # Ensure that some structural assumptions for self.basis hold
        # i.e. basis = [x1, x2, x3, xdot1, xdot2, xdot3]
        diff_order_list = [b.difforder for b in self.basis]
        do_max = max(diff_order_list)
        base_length = int(self.dim_basis / (do_max + 1)  )
        # base_length ≙ 3 in the above example

        expected_diff_order_list = [int(k/base_length) for k in range(self.dim_basis)]

        if not diff_order_list == expected_diff_order_list:
            msg = "the structure of the basis variables is not like expected. \n"
            msg += "Got:      %s\n" % diff_order_list
            msg += "Expected: %s\n" % expected_diff_order_list
            raise ValueError(msg)

        return base_length

    def dot(self, additional_symbols=None):
        """
        returns the time derivative of this n-form:

        self = a*dx + 0*dy + 0*dxdot + 0*dydot
        self.dot() == adot*dx + a*dxdot

        currently only supported for 1- and 2-forms

        additional_symbols is an optional list of time_dependent symbols
        """

        if not self.degree <= 2:
            raise NotImplementedError(".dot only tested for degree <= 2, might work however")

        base_length = self._calc_base_length()

        if additional_symbols is None:
            additional_symbols = []
        additional_symbols = list(additional_symbols)

        res = DifferentialForm(self.degree, self.basis)  # create a zero form

        # get nonzero coeffs and their indices
        nz_tups = [(idx_tup, c) for idx_tup, c in zip(self.indices, self.coeff) if c != 0]

        if len(nz_tups) == 0:
            # this form is already zero
            # -> return a copy
            return 0*self

        idx_tups, coeffs = lzip(*nz_tups)
        # idx_tups = e.g. [(1, 4), ...] (2-Form) or [(0,), (2,), ....] (1-Form)

        # nested list comprehension http://stackoverflow.com/a/952952/333403
        flat_nonzero_idcs = [i for idx_tup in idx_tups for i in idx_tup]

        # reduce duplicates
        flat_nonzero_idcs = list(set(flat_nonzero_idcs))
        flat_nonzero_idcs.sort()

        # get corresponding coords

        nz_coords = [self.basis[i] for i in flat_nonzero_idcs]
        nz_coords_diff = [st.time_deriv(c, self.basis) for c in nz_coords]

        # difference set
        ds = set(nz_coords_diff).difference(self.basis)
        if ds:
            msg = "The time derivative of this form cannot be calculated. "\
                  "The following necessary coordinates are not part of self.basis: %s" % ds
            raise ValueError(msg)

        # get new indices:
        basis_list = list(self.basis)
        #diff_idcs = [basis_list.index(c) for c in nz_coords_diff]

        # assume self = a*dx1^dx3
        # result: self.dot = adot*dx1^dx3 + a*dxdot1^dx3 + a*dx1^dxdot3

        # replace the original coeff with its time derivative (adot*dx1^dx3)
        for idcs, c in zip(idx_tups, coeffs):
            res[idcs] = st.time_deriv(c, basis_list + additional_symbols)

        # now for every coordinate find the basis-index of its derivative, e.g.
        # if basis is x1, x2, x3 the index of xdot2 is 4
        # set the original coeff to the corresponding place (a*dxdot1^dx3 + a*dx1^dxdot3)
        for idx_tup, c in zip(idx_tups, coeffs):
            for j, idx in enumerate(idx_tup):
                # convert to mutable data type (list instead of tuple)
                idx_list = list(idx_tup)
                idx_list[j] += base_length  # convert x3 to xdot3

                # add the coeff
                res[idx_list] += c

        return res

    def count_ops(self, *args, **kwargs):
        """Utility that returns a form of the same basis and degree as self
        where the coeffs are numbers corresponding to the application of count_ops to self.coeff.
        """

        coeff_co = st.count_ops(self.coeff, *args, **kwargs)
        return DifferentialForm(self.degree, self.basis, coeff=coeff_co)

    @property
    def co(self):
        return self.count_ops()

    @property
    def srn(self):
        return self.coeff.srn


def ensure_not_sympy_matrix_mul():
    """
    This auxiliary function prevents that the case M*V can be evaluated.
    (M: sympy Matrix, V: VectorDifferentialForm).

    The expression M*V triggers M.__mul__(V) to be called. Because V is
    not a Matrix it is handled as a scalar. This generate a unwanted result
    but no exception. To caluclate this unwanted result V is left-multiplied by
    every entry of M (-> V.__mul__). This method tries to detect that situation
    and raise a proper exception.

    :return: None
    """

    # TODO: this is not a clean solution (it should be solved on sympy side)

    cf = inspect.currentframe()
    fi1 = inspect.getframeinfo(cf.f_back.f_back).function
    fi2 = inspect.getframeinfo(cf.f_back.f_back.f_back).function

    if fi1 == "<listcomp>" and fi2 == "_eval_scalar_mul":
        # no message necessary because sympy will catch this and generate
        # its own error
        raise TypeError


class VectorDifferentialForm(CantSympify):
    def __init__(self, n, basis, coeff=None, basis_forms_str=None):
        self.degree = n
        self.basis = sp.Matrix(basis)
        self.basis_forms_str = basis_forms_str

        if not coeff==None:
            self.coeff = coeff
            self.m, self.n = coeff.shape
        else:
            self.m = self.n = 0
            self.coeff = sp.Matrix([])

        self._w = []
        for i in range(0, self.m):
            w1_coeffs = coeff.row(i)
            wi = DifferentialForm(self.degree, self.basis, coeff=w1_coeffs)
            self._w.append(wi)

    def __repr__(self):
        if self.basis_forms_str==None:
            basis_forms = "\"dX\""
        else:
            basis_forms = self.basis_forms_str
        return sp.sstr(self.coeff) + basis_forms

    def __add__(self, a):
        assert isinstance(a, VectorDifferentialForm)
        assert self.degree == a.degree
        assert self.basis == a.basis

        new_vector_form = VectorDifferentialForm(self.degree, self.basis)
        new_vector_form.coeff = self.coeff + a.coeff

        return new_vector_form

    def __mul__(self, a):
        """
        called if self is the left operand.

        :param a:
        :return:
        """
        ensure_not_sympy_matrix_mul()
        if not st.is_scalar(a):
            msg = "Multiplication of %s and %s not (currently) not allowed. " \
                  "Maybe use .left_mul_by(...)."
            msg = msg %(type(self), type(a))
            raise TypeError(msg)

        new_vector_form = VectorDifferentialForm(self.degree, self.basis)
        new_vector_form.coeff = nct.nc_mul(self.coeff, a)

        return new_vector_form

    def __rmul__(self, a):
        """
        called if self is the right operand and the __mul__ method of the left was
        absent or raised NotImplementedError.

        :param a:
        :return:
        """
        if not st.is_scalar(a):
            msg = "Reverse multiplication of %s and %s not (currently) not allowed. " \
                  "Maybe use .left_mul_by(...)."
            msg = msg %(type(self), type(a))
            raise TypeError(msg)

        new_vector_form = VectorDifferentialForm(self.degree, self.basis)
        new_vector_form.coeff = nct.nc_mul(a, self.coeff)

        return new_vector_form

    def __sub__(self, m):
        neg_m = VectorDifferentialForm(m.degree, m.basis, -1*m.coeff)
        return self + neg_m

    # TODO: Unit test
    def __getitem__(self, ind):
        return self._w.__getitem__(ind)

    @property
    def srn(self):
        return self.coeff.srn

    def simplify(self, *args, **kwargs):
        self.coeff.simplify(*args, **kwargs)
        for i in range(0, self.m):
            self._w[i].coeff.simplify(*args, **kwargs)

    def left_mul_by(self, matrix, s=None, additional_symbols=None):
        """ Performs matrix*vectorform and returns the new vectorform.
            additional_symbols is an optional list of time_dependent symbols
            Note: matrix is assumed to be polynomial in s
        """
        assert isinstance(matrix, sp.MatrixBase)

        m1, n1 = matrix.shape
        assert n1==self.m, "Matrix Dimesion does not fit vector form!"

        # right shift s:
        matrix_shifted = nct.right_shift_all(matrix, s=s, func_symbols=additional_symbols)

        # if s is of higher order, raise not implemented
        if not s==None and matrix_shifted.diff(s).has(s):
            raise NotImplementedError

        if s==None:
            M0 = matrix_shifted
        else:
            M1 = matrix_shifted.diff(s)
            M0 = matrix_shifted - M1*s

        new_vector_form = VectorDifferentialForm(self.degree, self.basis)
        for i in range(0, m1):
            new_wi = DifferentialForm(1, self.basis)
            for j in range(0, n1):
                new_wi += M0[i,j] * self.get_differential_form(j)
                if not s==None:
                    new_wi += M1[i,j] * self.get_differential_form(j).dot(additional_symbols)
            new_vector_form.append(new_wi)

        return new_vector_form

    def dot(self):
        new_vector_form = VectorDifferentialForm(self.degree, self.basis)
        for i in range(0, self.m):
            new_wi = self.get_differential_form(i).dot()
            new_vector_form.append(new_wi)
        return new_vector_form

    def unpack(self):
        ww = []
        for i in range(0, self.m):
            ww.append(self._w[i])
        return tuple(ww)

    def get_differential_form(self, i):
        return self._w[i]

    def get_coeff_from_idcs(self, idcs):
        return self.coeff.col(idcs)

    def append(self, k_form):
        assert k_form.degree==self.degree, "Degrees of vector forms do not match."

        if isinstance(k_form, DifferentialForm):
            rows = k_form.coeff.T
        else:
            rows = k_form.coeff
        self._w.append(k_form)
        self.coeff = st.row_stack(self.coeff, rows)
        self.m = self.m + 1

    def subs(self, *args, **kwargs):
        matrix = self.coeff.subs(*args, **kwargs)
        new_vector_form = VectorDifferentialForm(self.degree, self.basis, coeff=matrix)
        return new_vector_form

def stack_to_vector_form(*arg):
    """
    This function stacks k-forms to a vector k-form.
    """
    # TODO: asserts

    if len(arg)==1:
        return arg[0]
    else:
        XX = arg[0].basis
        k = arg[0].degree

        coeff_matrix = sp.Matrix([])
        for i in range(0, len(arg)):
            coeff_matrix_i = arg[i].coeff.T
            coeff_matrix = st.row_stack(coeff_matrix, coeff_matrix_i)

        return VectorDifferentialForm(k, XX, coeff=coeff_matrix)

def coeff_ido_derivorder(sigma, *factors, **kwargs):
    """
    Calulate the coefficient corresponding to the N-tuple sigma of the N-form Mu_k
    in dependence of the derivative order k, where Mu_k denotes
    mu_1_(k)^mu_2_(k)^...^mu_N_(k) and mu_i_(k) denotes the k-th
    time-derivative of the 1-form mu_i.

    :param sigma:       index tuple (length N)
    :param factors:     n-tuple of 1-forms (mu1, mu2, ...)
    :return:            special coeff (see description)

    optional parameters:
    :do_symbol:         symbol for the derivative order in the result
    :tds:               time dependend symbols (used for mu_i.dot(tds) )


    How this is done:
    Step 1:
    Replace every entry in mu_i.coeff by a unique symbol S_j (dependent on time) and
    denote the resulting 1-form by zeta_i.

    Step 2:
    For every i in {1, ... , n} calc zeta_i.dot().

    Step 3:
    Calc Z := zeta_1.dot()^zeta_2.dot()^...^zeta_N.dot().


    Step 4:
    # TODO: this is obsolete:
    Assume that for all mu_i we have something like: mu.basis = [x1, ..., xn, xdot1, ....]
    (i.e. n is the number of basis-variables x1, ..., xn, which occur in
    several derivative orders).
    Extract the coefficient "c_res0" of Z, corresponding to sigma (elementwise addition).

    Step 5:
    In c_res0 replace the symbols "S_dot" by k*S_dot.
    Denote the resulting expression by c_res1.

    Step 6:
    In c_res1 replace all symbols in S and S_dot by their corresponding
    expression from the original 1-forms mu_i. Denote the result by c_res2

    Return c_res(2)

    Note: The result is only correct for some coefficients.
    In all other cases a ValueError is raised -> Step 0:
    """

    assert len(sigma) == len(factors) > 0

    tds = kwargs.get('tds', [])
    tds = list(tds)

    basis = factors[0].basis

    # Step 0 (and consistency checking):
    # determine the derivative orders of the basis-symbols
    deriv_orders = [bs.difforder for bs in basis]
    min_do = min(deriv_orders)  # usually will be 0
    max_do = max(deriv_orders)

    msg = "This function assumes more than one derivative order!"
    assert (min_do + 1) in deriv_orders, msg
    n = deriv_orders.index(min_do + 1)
    assert len(deriv_orders) % n == 0
    #groups = []
    for i in range(int(len(deriv_orders)/n)):
        tmp_group = deriv_orders[n*i: n*(i+1)]
        msg2 = "All group elements are expected to be identical"
        assert all( elt == tmp_group[0] for elt in tmp_group), msg2

    msg3 = "Unexpected distribution of derivorders (like e.g. [0,0,2,2]): "
    msg3 += str(deriv_orders)
    assert (max_do - min_do + 1)*n == len(deriv_orders), msg3

    if isinstance(sigma, np.ndarray):
        sigma = tuple(int(idx) for idx in sigma)
    assert sigma == tuple(sorted(sigma))

    extended_deriv_orders = deriv_orders + [max_do + 1]*n
    deriv_orders_of_sigma = []

    for idx in sigma:
        if idx >= len(extended_deriv_orders):
            msg4 = "Uexpected high index in sigma: %s." % str(sigma)
            raise ValueError(msg4)

        deriv_orders_of_sigma.append( extended_deriv_orders[idx] )

    # do we need to call jet_extend_basis onb the factors?
    cond1 = all( idx in (max_do, max_do - 1) for idx in deriv_orders_of_sigma)
    cond2 = all( idx in (max_do + 1, max_do) for idx in deriv_orders_of_sigma)

    if cond1:
        jeb_flag = False
    elif cond2:
        jeb_flag = True
    else:
        msg5 = "Only indices which correspond to the highest or second highest "\
               "deriv_order are allowed in sigma."
        raise ValueError(msg5)

    zeta_dot_list = []
    gen_S = sp.numbered_symbols('S')

    replacements_S = []
    S_list = []
    Z = 1  # 0-form

    # Step 1 and Step 2 and Step 3:

    for i, mu in enumerate(factors):

        assert isinstance(mu, DifferentialForm), "not a form: %i" % i
        assert mu.degree == 1
        assert mu.basis == basis

        # -> Step 1:
        zeta = mu*0  # make an empty 1-form

        if jeb_flag:
            zeta.jet_extend_basis()
            mu = mu*1 # make a copy
            mu.jet_extend_basis()

        for i, c in enumerate(mu.coeff):
            if st.is_number(c):
                zeta.coeff[i] = c
                continue

            S_tmp = next(gen_S)
            replacements_S.append( (S_tmp, c) )
            S_list.append(S_tmp)
            zeta.coeff[i] = S_tmp

        # -> Step 2
        zeta_dot = zeta.dot(S_list)
        zeta_dot_list.append(zeta_dot)

        # -> Step 3 (recursively)
        Z = Z*zeta_dot

    # Step 4:
    #sigma_plus_nn = tuple((elt + n for elt in sigma))

    c_res0 = Z.get_coeff_from_idcs(sigma)

    # Step 5:
    default_do_symbol = sp.Symbol('l')
    do_symbol = kwargs.get('do_symbol', default_do_symbol)
    S_vector = sp.Matrix(S_list)
    S_dot_vector = st.time_deriv(S_vector, S_vector)

    c_res1 = c_res0.subs(zip(S_dot_vector, do_symbol*S_dot_vector))

    # Step 6
    # constructing the whole replacement structure
    S_repl_matrix = sp.Matrix(replacements_S)  # two columns: symbols, expressions
    tds2 = tds + S_list + list(basis)
    S_repl_matrix_dot = st.time_deriv(S_repl_matrix, tds2)

    all_replmts = S_repl_matrix_dot.tolist() + replacements_S

    c_res2 = c_res1.subs(all_replmts)

    return c_res2, do_symbol


def pull_back(phi, args, omega):
    """
    computes the pullback phi^* omega for a given mapping phi between
    manifolds (assumed to be given as a 1-column matrix
    (however, phi is not a vector field))
    """

    assert isinstance(phi, sp.Matrix)
    assert phi.shape[1] == 1
    assert phi.shape[0] == omega.dim_basis
    assert isinstance(omega, DifferentialForm)

    if omega.grad > 1:
        raise NotImplementedError("Not yet implemented")

    # the arguments of phi are the new basis symbols

    subs_trafo = lzip(omega.basis, phi)
    new_coeffs = omega.coeff.T.subs(subs_trafo) * phi.jacobian(args)

    res = DifferentialForm(omega.grad, args, coeff=new_coeffs)

    return res


def _contract_vf_with_basis_form(vf, idx_tuple):
    """
    calculate (v˩A) where v is a vector field and A is a
    basisform (like dx0^dx3^dx4) which is represented by
    an index tuple (like (0,3,4))

    returns a list of result tuples (coeff, idcs)
    where coeff is the resulting coefficient and idcs are the
    corresponding remaining indices

    example: v = sp.Matrix([0, 0, 0, v3, v4])
    return [(0, (3,4)), (-v3 (0,4)), (v4, (0,3))]
    """

    # Algorithm: idx_tuple represents a basis k-form
    # removing one of the indices of that tuple represents a basis (k-1)-form
    # on the other hand, removing the i-th index and multiplying by (-1)**i
    # corresponds to factoring out (w.r.t. ⊗) the basis 1-form corresponding
    # to that index:

    # x^y^z = x⊗y⊗z-x⊗z⊗y + y⊗z⊗x-y⊗x⊗z + z⊗x⊗y-z⊗y⊗x
    #       = x⊗(y^z) - y⊗(x^z) + z⊗(x^y)

    # for details, see Spivak Calculus on Manifolds, Chapter 4
    # Note that, here we use the convention
    # x^y = (k+l)!/(k!*l!) * Alt(x⊗y)
    # Some authors omit the leading factor

    assert vf.shape[1] == 1 and vf.shape[0] > max(idx_tuple)
    res = []
    for i, idx in enumerate(idx_tuple):
        sgn = (-1) ** i
        # determine the remaining indices
        rest_idcs = list(idx_tuple)
        rest_idcs.pop(i)
        rest_idcs = tuple(rest_idcs)

        res.append((sgn * vf[idx], rest_idcs))
    return res


#TODO: unit tests (yet only checked for 2- and 3-forms)
def contraction(vf, form):
    """
    performs the interior product of a vector field v and a p-form A
    (v˩A)(x,y,z,...) = A(v, x, y, z, ...)

    expects vf (v) as a column Matrix (coefficients of suitable basis vectors)
    """

    assert isinstance(vf, sp.Matrix)
    n1, n2 = vf.shape
    assert n1 == form.dim_basis and n2 == 1

    if form.grad == 0:
        return 0  # by definition

    # A can be considered as a sum of decomposable basis-forms
    # contraction can be performed for each of the basis components
    # A.indices contains information about the basis-Vectors
    # we only need those index-tuples with a nonzero coefficient

    nzt = form.nonzero_tuples()

    # example: A = c1*dx0^dx1 + c2*dx2^dx4 + ...
    coeffs, index_tuples = lzip(*nzt)  # -> [(c1, c2, ...), ((0,1), (2, 4), ...)]

    # our result will go there
    result = DifferentialForm(form.grad - 1, form.basis)

    for coeff, idx_tup in nzt:
        part_res = _contract_vf_with_basis_form(vf, idx_tup)
        for c, rest_idcs in part_res:
            # each rest-index-tuple can occur multiple times -> sum entries
            tmp = result.__getitem__(rest_idcs) + c * coeff
            result.__setitem__(rest_idcs, tmp)

    return result

def simplify(arg, **kwargs):
    """
    Simplification Function which is aware of (Vector)DifferentialForms
    """

    if isinstance(arg, (DifferentialForm, VectorDifferentialForm)):
        copy = arg*1
        copy.simplify(**kwargs)
        return copy
    else:
        return sp.simplify(arg, **kwargs)



# Keilprodukt zweier Differentialformen
def keilprodukt(df1, df2):
    # Todo: hier kam mal irgendwan fälschlicherweise 0 raus
    # siehe meine Mail an  Torsten vom 03.03.2014.

    # Funktion zum Prüfen, ob zwei Tupel identische Einträge enthalten
    def areTuplesAdjunct(tpl1, tpl2):
        for n in tpl1:
            if (n in tpl2):
                return False
        return True

    res = DifferentialForm(df1.grad + df2.grad, df1.basis)
    for n in range(df1.num_coeff):
        if df1.grad == 0:
            df1_n = ()
        else:
            df1_n = df1.indizes[n]
        for m in range(df2.num_coeff):
            if df2.grad == 0:
                df2_m = ()
            else:
                df2_m = df2.indizes[m]
            if areTuplesAdjunct(df1_n, df2_m):
                res[df1_n + df2_m] += df1.coeff[n] * df2.coeff[m]
    return res


def wp2(*args):
    """wedge product"""
    return reduce(keilprodukt, args)


def wp(a, b, *args):
    assert a.basis == b.basis
    N = len(a.basis)
    DEG = a.grad + b.grad

    name = "%s^%s" % (a.name, b.name)
    res = DifferentialForm(DEG, a.basis, name=name)

    new_base_tuples = list(it.combinations(list(range(N)), DEG))

    NZ_a = a.nonzero_tuples()  # list of tuples: (coeff, indices)
    NZ_b = b.nonzero_tuples()

    # cartesian product (everyone with everyone)
    Prod = it.product(NZ_a, NZ_b)  #[(A, B), ...],A=(ca1, idcs1), B = analogical

    for (ca, ia), (cb, ib) in Prod:
        if not set(ia).intersection(ib):
            i_new = ia + ib
            s = perm_parity(i_new)

            i_new = tuple(sorted(i_new))
            basis_index = new_base_tuples.index(i_new)
            res.coeff[basis_index] += s * ca * cb

    if args:
        res = wp(res, args[0], *args[1:])
    return res


def basis_1forms(basis):
    """
    basis_1forms((x1, x2, u, t)) -> dx1, dx2, du, dt
    """
    N = len(basis)
    z = sp.zeros(N, 1)
    res = []
    for i in range(N):
        tmp = z * 1
        tmp[i, 0] = 1
        name = "d" + basis[i].name
        res.append(DifferentialForm(1, basis, coeff=tmp, name=name))

    return res


# todo: this function should be callable with Differential forms also
def d(func, basis):
    return DifferentialForm(0, basis, coeff=[func]).d


def setup_objects(n):
    """
    convenience function
    creates a coordinate basis, and a set of canonical one-forms and
    inserts both in the global name space
    """

    if isinstance(n, int):
        xx = sp.symbols("x1:%i" % (n + 1))
    elif isinstance(n, string_types):
        xx = sp.symbols(n)

    # now assume n is a sequence of symbols
    elif all([x.is_Symbol for x in n]):
        xx = n
    else:
        raise TypeError("unexpected argument-type: " + str(type(n)))

    bf = basis_1forms(xx)

    st.make_global(xx, upcount=2)
    st.make_global(bf, upcount=2)

    return xx, bf

# for backward compatibility
diffgeo_setup = setup_objects


def _test_pull_back_to_sphere():
    """
    pull back the differential of the function F = (y1**2 + y2**2 + y3**2)
    it must vanish on the sphere. Hereby we use spherical coordinates
    to consider the sphere as immersed submanifold of R3
    """

    yy = y1, y2, y3 = sp.symbols("y1:4")
    xx = x1, x2 = sp.symbols("x1:3")

    F = y1 ** 2 + y2 ** 2 + y3 ** 2
    omega = dF = d(F, yy)
    # spherical coordinates:
    from sympy import sin, cos

    phi = sp.Matrix([cos(x1) * cos(x2), sin(x1) * cos(x2), sin(x2)])

    p = pull_back(phi, xx, omega)
    assert p.is_zero()


def _main():
    """
    for testing
    """
    from IPython import embed as IPS

    xx = sp.symbols("x1:6")
    dxx = basis_1forms(xx)

    dx1, dx2, dx3, dx4, dx5 = dxx
    t = wp(dx1, xx[4] * dx2)

    IPS()


if __name__ == "__main__":
    _test_pull_back_to_sphere()
    #_main()
