# -*- coding: utf-8 -*-
"""
Created on Tue Sep 03 17:44:46 2013

@author: Torsten Knüppel (original Code)
@author: Carsten Knoll (enhancements)
"""

from sympy import Function, diff
from sympy.matrices import zeros
from itertools import combinations
import numpy as np
import sympy as sp
import itertools as it

#from ipHelp import IPS

## Vorzeichen einer Permutation
# original-Algorithmus -> liefert manchmal 0 bei längeren Permutationen
def sign_perm(perm):
    perm = range_indices(perm)
    Np = len(np.array(perm))
    sgn = 1
    for n in range(Np - 1):
        for m in range(n + 1,Np):
            sgn *= 1. * (perm[m]  - perm[n]) / (m - n)
    sgn = int(sgn)
#    if not sgn in (-1, 1):
#        IPS()
    assert sgn in (-1, 1), "error %s" %perm

    return sgn
#
#def perm_parity2(seq):
#
#    L = np.arange(len(seq)) % 2 # array r_[0, 1, 0, 1, ...]
#
#    tuples = zip(seq, L)
#    tuples.sort(key = lambda x: x[0])
#
#    bin_info = np.array(zip(*tuples)[1])
#
#
#    IPS()

def range_indices(seq):
    """
    returns a tuple which represents the same permutation
    but with indices from range(N)
    (5, 2, 10, 1) -> (2, 1, 3, 0)
    """
    assert len(set(seq)) == len(seq)

    #assert False
    # range_indices((0, 1, 3, 8, 2, 4))

    new_elements = range(len(seq))

    res = [None]* len(seq)

    seq = list(seq)
    seq_work = list(seq)

    for i in range(len(seq)):
        m1 = min(seq_work)
        m2 = min(new_elements)

        i = seq.index(m1)
        res[i] = m2

        seq_work.remove(m1)
        new_elements.remove(m2)

#    tuples = list(enumerate(seq))
#    tuples.sort(key = lambda x: x[1])
#    res = zip(*tuples)[0]
    assert not None in res
    return res


def perm_parity(seq):
    '''\
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    '''
    # adapted from http://code.activestate.com/
    # recipes/578227-generate-the-parity-or-sign-of-a-permutation/
    # by Paddy McCarthy
    lst = range_indices(seq) # normalize the sequence to the first N integers
    parity = 1
    index_list = range(len(lst))
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            # there must be a smaller number in the remaining list
            # -> perform an exchange
#            print i, lst
            mn = np.argmin(lst[i:]) + i
            lst[i],lst[mn] = lst[mn],lst[i]
#            print mn, lst
            parity *= -1
    return parity


class DifferentialForm:
    def __init__(self, n, basis, koeff=None, name=None):
        """
        :n: degree
        :basis: list of basis coordinates (Symbols)

        """
        # Grad der Differentialform
        self.grad = n
        # Basis der Differentialform
        self.basis = basis
        # Dimension der Basis
        self.dim_basis = len(basis)
        # Liste zulässiger Indizes
        if(self.grad == 0):
            self.indizes = [(0,)]
        else:
            #TODO: this should be renamed to index_tuples
            self.indizes = list(combinations(range(self.dim_basis),self.grad))


        # Anzahl der Koeffizienten der Differentialform
        self.num_koeff = len(self.indizes)
        # Koeffizienten der Differentialform
        if koeff == None:
            self.koeff = zeros(self.num_koeff,1)
        else:
            assert len(koeff) == self.num_koeff
            self.koeff = sp.Matrix(koeff).reshape(self.num_koeff, 1)

        self.name = name # usefull for symb_tools.make_global


    # TODO: the property should be named coeff instead of coeff
    # quick hack combine new name with backward compability
    @property
    def coeff(self):
        return self.koeff


    def __repr__(self):
        return self.ausgabe()

    def __add__(self, a):


        assert self.grad == a.grad
        assert self.basis == a.basis

        new_form = DifferentialForm(self.grad, self.basis)
        new_form.koeff = self.koeff + a.koeff

        return new_form


    def __sub__(self, m):

        return self + (m*(-1))


    def __rmul__(self, f):
        """scalar multiplication from the left (reverse mul)"""
        # use commutativity
        return self*f


    def __mul__(self, f):
        """ scalar multipplication"""
        new_form = DifferentialForm(self.grad, self.basis)

        new_form.koeff = self.koeff*f
        return new_form

    def __eq__(self, other):
        """test for equality"""
        assert isinstance(other, DifferentialForm)
        assert self.basis == other.basis
        if self.grad == other.grad:
            return self.koeff == other.koeff
        else: return False


    def __neg__(self):
        return -1*self

    def __getitem__(self,ind):
        """Koeffizient der Differentialform zuweisen"""
        assert len(ind) == self.grad
        try:
            ind_1d, vz = self.__getindexperm__(ind)
        except ValueError:
            print u'invalid Index'
            return None
        else:
            return vz * self.koeff[ind_1d]

    def __setitem__(self,ind,wert):
        """Koeffizient der Differentialform abrufen"""
        assert len(ind) == self.grad
        try:
            ind_1d, vz = self.__getindexperm__(ind)
        except ValueError:
            print u'invalid Index'
        else:
            self.koeff[ind_1d] = vz * wert

    def __getindexperm__(self,ind):
        """ Liefert den Index und das Vorzeichen der Permutation"""
        ind = np.atleast_1d(ind)
        if(len(ind) == 1):
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
        # Neue Differentialform mit selber Basis erstellen
        res = DifferentialForm(self.grad + 1,self.basis)
        # 0-Form separat
        if(self.grad == 0):
            for m in range(self.dim_basis):
                res[m] = diff(self.koeff[0],self.basis[m])
        else:
            for n in range(self.num_koeff):
                ind_n = self.indizes[n]
                for m in range(self.dim_basis):
                    if(m in ind_n):
                        continue
                    else:
                        res[(m,) + ind_n] += diff(self.koeff[n],self.basis[m])

        return res

    def nonzero_tuples(self):
        """
        returns a list of tuples (coeff, idcs) for each index tuple idcs,
        where the corresponding coeff == 0
        """
        Z = zip(self.koeff, self.indizes)
        res = [(c, idcs) for (c, idcs) in Z if c != 0 ]

        return res

    def is_zero(self, n = 100):
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

    @property
    def d(self):
        """property -> allows for short syntax: w1.d (= w1.diff())"""
        return self.diff()

    def subs(self, *args, **kwargs):
        """
        returns a copy of this form with subs(...) applied to its koeff-matrix
        """
        res = DifferentialForm(self.grad, self.basis)
        res.koeff = self.koeff.subs(*args, **kwargs)
        return res

    def change_of_basis(self, new_basis, Psi, Psi_jac = None):
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
        assert len(new_basis) == len (self.basis)
        assert self.grad <= 1 # not yet understood for higher order
        res = DifferentialForm(self.grad, new_basis)
        k = self.koeff.subs( zip(self.basis, Psi) )

        if Psi_jac == None:
            Psi_jac = Psi.jacobian(new_basis)

        k_new = (k.T*Psi_jac).T
        assert k_new.shape == (len(new_basis), 1)

        res.koeff = k_new
        return res

# TODO doctest
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
        res = self[idcs]/coeff

        return res



    # Differentialform ausgeben
    def ausgabe(self):
        # 0-Form separat
        if(self.grad == 0):
            df_str = str(self.koeff[0])
        else:
            df_str = '0'
            for n in range(self.num_koeff):
                koeff_n = self.koeff[n]
                ind_n = self.indizes[n]
                if(koeff_n == 0):
                    continue
                else:
                    # String des Koeffizienten
                    sub_str = '(' + self.eliminiere_Ableitungen(self.koeff[n]) + ') '
                    # Füge Basis-Vektoren hinzu
                    for m in range(self.grad - 1):
                        sub_str += 'd' + (self.basis[ind_n[m]]).name + '^'
                    sub_str += 'd' + (self.basis[ind_n[self.grad - 1]]).name
                # Gesamtstring
                if(df_str == '0'):
                    df_str = sub_str
                else:
                    df_str += '+' + sub_str
        # Ausgabe
        return df_str

    def eliminiere_Ableitungen(self,koeff):
        koeff = sp.simplify(koeff)
        at_deri = list(koeff.atoms(sp.Derivative))
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
                dfunc = Function('D' + str(ndiff) + name_diff)(*diff_arg)
                # Neue Funktion einsetzen und Substitutionen durchführen
                koeff = koeff.subs(deri_n,dfunc).doit()
        return str(koeff)

    # TODO: unit test, extend to higher degrees
    def integrate(self):
        """
        assumes self to be a 1-Form
        if self.d.is_zero() then this method returns h such that:
            d h = self
        """

        if not self.grad == 1: raise NotImplementedError("not yet supported")
        assert self.d.is_zero()

        res = 0
        for b,c in zip(self.basis, self.coeff):
            res += sp.integrate(c, b)

        return res

    def contract(self, vf):
        """
        Contract this differential form with a vectorfield
        (appply the form on it)

        We assume the same basis

        currently only implemented for 1-forms
        """

        if not self.grad == 1: raise NotImplementedError("not yet supported")
        assert vf.shape == self.coeff.shape
        res = self.coeff.T*vf
        return res[0,0]

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
        test_form = wp(ds, s) # contains (ds)**(r+1) ^ s
        while True:
            test_form.coeff.simplify()

            if test_form.coeff == test_form.coeff*0:
                return r

            r += 1
            test_form = wp(ds, test_form)

            assert r <= (self.dim_basis-1)/2.0



def _contract_vf_with_basis_form(vf, idx_tuple):
    """
    calculate (v˩A) where A is a basisform (like dx0^dx3^dx4) which
    is represented by an index tuple (like (0,3,4))

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

    assert vf.shape[1] == 1 and vf.shape[0] > max(idx_tuple)
    res = []
    for i, idx in enumerate(idx_tuple):
        sgn = (-1)**i
        # determine the remaining indices
        rest_idcs = list(idx_tuple)
        rest_idcs.pop(i)
        rest_idcs = tuple(rest_idcs)

        res.append( (sgn*vf[idx], rest_idcs) )
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
        return 0 # by definition

    # A can be considered as a sum of decomposable basis-forms
    # contraction can be performed for each of the basis components
    # A.indices contains information about the basis-Vectors
    # we only need those index-tuples with a nonzero coefficient

    nzt = form.nonzero_tuples()

    # example: A = c1*dx0^dx1 + c2*dx2^dx4 + ...
    coeffs, index_tuples = zip(*nzt) # -> [(c1, c2, ...), ((0,1), (2, 4), ...)]

    # our result will go there
    result = DifferentialForm(form.grad-1, form.basis)

    for coeff, idx_tup in nzt:
        part_res = _contract_vf_with_basis_form(vf, idx_tup)
        for c, rest_idcs in part_res:
            # each rest-index-tuple can occur multiple times -> sum entries
            tmp = result.__getitem__(rest_idcs) + c*coeff
            result.__setitem__(rest_idcs, tmp)

    return result




# Keilprodukt zweier Differentialformen
def keilprodukt(df1,df2):
    # Todo: hier kam mal irgendwan fälschlicherweise 0 raus
    # siehe meine Mail an  Torsten vom 03.03.2014.

    # Funktion zum Prüfen, ob zwei Tupel identische Einträge enthalten
    def areTuplesAdjunct(tpl1,tpl2):
        for n in tpl1:
            if(n in tpl2):
                return False
        return True
    res = DifferentialForm(df1.grad + df2.grad, df1.basis)
    for n in range(df1.num_koeff):
        if(df1.grad == 0):
            df1_n = ()
        else:
            df1_n = df1.indizes[n]
        for m in range(df2.num_koeff):
            if(df2.grad == 0):
                df2_m = ()
            else:
                df2_m = df2.indizes[m]
            if(areTuplesAdjunct(df1_n,df2_m)):
                res[df1_n + df2_m] += df1.koeff[n] * df2.koeff[m]
    return res

def wp2(*args):
    """wedge product"""
    return reduce(keilprodukt, args)


def wp(a, b, *args):
    assert a.basis == b.basis
    N = len(a.basis)
    DEG = a.grad+b.grad

    name = "%s^%s" % (a.name, b.name)
    res = DifferentialForm(DEG, a.basis, name=name)

    new_base_tuples = list(it.combinations(range(N), DEG))

    NZ_a = a.nonzero_tuples() # list of tuples: (coeff, indices)
    NZ_b = b.nonzero_tuples()

    # cartesian product (everyone with everyone)
    Prod = it.product(NZ_a, NZ_b) #[(A, B), ...],A=(ca1, idcs1), B = analogical

    for (ca, ia), (cb, ib) in Prod:
        if not set(ia).intersection(ib):
            i_new = ia+ib
            s = perm_parity(i_new)

            i_new = tuple(sorted(i_new))
            basis_index = new_base_tuples.index(i_new)
            res.koeff[basis_index] += s*ca*cb


    if args:
        res = wp(res, args[0], *args[1:])
    return res



def basis_1forms(basis):
    """
    basis_1forms((x1, x2, u, t)) -> dx1, dx2, du, dt
    """
    N = len(basis)
    z = sp.zeros(N,1)
    res = []
    for i in range(N):
        tmp = z*1
        tmp[i,0] = 1
        name = "d"+basis[i].name
        res.append(DifferentialForm(1, basis, koeff=tmp, name=name))

    return res


def d(func, basis):
    return DifferentialForm(0, basis, koeff=[func]).d


def diffgeo_setup(n):
    """
    convenience function
    creates a coordinate basis, and a set of canonical one-forms and
    inserts both in the global name space
    """

    xx = sp.symbols("x1:%i" % (n+1))
    bf = basis_1forms(xx)

    import symb_tools as st

    st.make_global(xx, up_count = 2)
    st.make_global(bf, up_count = 2)

    return xx, bf



"""
Schnipsel für Unit-Tests:

    In [5]: (x1, x2, r), (dx1, dx2, dr)= dg.diffgeo_setup(3)

    In [6]: aa = a*dr-r*dx1-l*dx2

    In [7]: aa.rank()
    Out[7]: 1



"""



def _main():
    """
    for testing
    """
    from ipHelp import IPS, ip_syshook
    xx = sp.symbols("x1:6")
    dxx = basis_1forms(xx)

    dx1, dx2, dx3, dx4, dx5 = dxx
    t = wp(dx1, xx[4]*dx2)

    IPS()

if __name__ == "__main__":
    _main()
