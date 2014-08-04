# -*- coding: utf-8 -*-
"""
Created on Tue Sep 03 17:44:46 2013

@author: knueppel
"""

from sympy import Function, symbols, diff
from sympy.matrices import zeros
from itertools import combinations
import numpy as np
import sympy as sp

# Vorzeichen einer Permutation
def sign_perm(perm):
    Np = len(np.array(perm))
    sgn = 1
    for n in range(Np - 1):
        for m in range(n + 1,Np):
           sgn *= 1. * (perm[m]  - perm[n]) / (m - n)
    return int(sgn)

class DifferentialForm:
    def __init__(self,n,basis):
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
            self.indizes = list(combinations(range(self.dim_basis),self.grad))        
        # Anzahl der Koeffizienten der Differentialform
        self.num_koeff = len(self.indizes)
        # Koeffizienten der Differentialform
        self.koeff = zeros(self.num_koeff,1)
        
    
    # Koeffizient der Differentialform zuweisen
    def __getitem__(self,ind):
        try:        
            ind_1d, vz = self.__getindexperm__(ind)
        except ValueError:
            print u'Ungültiger Index'
            return None
        else:
            return vz * self.koeff[ind_1d]
    
    # Koeffizient der Differentialform abrufen
    def __setitem__(self,ind,wert):
        try:
            ind_1d, vz = self.__getindexperm__(ind)
        except ValueError:
            print u'Ungültiger Index'
        else:
            self.koeff[ind_1d] = vz * wert
    
    # Liefert den Index und das Vorzeichen der Permutation
    def __getindexperm__(self,ind):
        ind = np.atleast_1d(ind)
        if(len(ind) == 1):
            ind_1d = self.indizes.index(tuple(ind))
            vz = 1
        else:
            srt_arr = np.argsort(ind)
            vz = sign_perm(srt_arr)
            ind_1d = self.indizes.index(tuple(ind[srt_arr]))
        return ind_1d, vz
    
    # Liefert die äußere Ableitung der Differentialform   
    def diff(self):
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
                elif(sub_str[0] == "-"):
                    df_str += sub_str
                else:
                    df_str += '+' + sub_str
        # Ausgabe
        print df_str
        
    def eliminiere_Ableitungen(self,koeff):
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
                x = sp.Symbol('x')
                dfunc = Function('D' + str(ndiff) + name_diff)(x)
                dfunc._args = diff_arg
                # Neue Funktion einsetzen und Substitutionen durchführen
                koeff = koeff.subs(deri_n,dfunc).doit()
        return str(koeff)        
                    
# Keilprodukt zweier Differentialformen
def keilprodukt(df1,df2):
    # Funktion zum Prüfen, ob zwei Tupel identische Einträge enthalten
    def areTuplesAdjunct(tpl1,tpl2):
        for n in tpl1:
            if(n in tpl2):
                return False
        return True
    res = DifferentialForm(df1.grad + df2.grad, df1.basis)
    for n in range(df1.num_koeff):
        for m in range(df2.num_koeff):
            df1_n = df1.indizes[n]
            df2_m = df2.indizes[m]
            if(areTuplesAdjunct(df1_n,df2_m)):
                res[df1_n + df2_m] += df1.koeff[n] * df2.koeff[m]
    return res
