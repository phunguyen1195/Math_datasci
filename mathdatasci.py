import numpy as np
from scipy import linalg
from discreteMarkovChain import markovChain
from sympy import *
import copy


class mathdatasci:
    def __init__(self, P, l, auto_l = True):
        self.P = P
        self.l = l
        self.v = np.empty(0)
        self.u = np.empty(0)
        self.Pl = np.empty(0)
        self.Pl_normalize_stochastic = np.empty(0)
        self.eigen_stationarydist_Pl = {}
        self.eigen_stationarydist_normalize_Pl = {}
        self.auto_l = auto_l
        if self.auto_l == True:
            self.get_v()
            self.get_u()
            self.get_Pnl()
        else:
            self.v = l
            self.get_u()
            self.get_Pnl()
        self.normalize_stochastic_matrix()
        
    
    def get_v (self):
        v = np.zeros((self.l.size, self.l.itemsize))
        for k in range(self.l.size):
            for o in self.l[k]:
                v[k][o] = 1.0
        v = v.T
        self.v = v

    def get_u (self):
        u = np.linalg.inv(np.dot(self.v.T,self.v))
        u = u.dot(self.v.T)
        self.u = u

    def get_Pnl (self):
        Pl = self.u.dot(self.P).dot(self.v)
        self.Pl = Pl

    def normalize_stochastic_matrix (self):
        Pl_normalize_stochastic = copy.deepcopy(self.Pl)
        for i in range (int (np.sqrt(Pl_normalize_stochastic.size))):
            k = 1 - Pl_normalize_stochastic[i][i]
            Pl_normalize_stochastic[i][i] = 0
            Pl_normalize_stochastic[i]  =Pl_normalize_stochastic[i]* ((k)**(-1))
        self.Pl_normalize_stochastic = Pl_normalize_stochastic

    def get_eigen_stationarydist_normalize_Pl(self):
        w, vl, vr = linalg.eig(self.Pl_normalize_stochastic, left=True)
        mc = markovChain(self.Pl_normalize_stochastic)
        mc.computePi('linear')
        self.eigen_stationarydist_normalize_Pl =  {'eigen values': np.around(w, 3), 'eigen vector': np.around(vr, 3),'stationary distribution': np.around(mc.pi,3)}
        return self.eigen_stationarydist_normalize_Pl

    def get_eigen_stationarydist_Pl(self):
        w, vl, vr = linalg.eig(self.Pl, left=True)
        mc = markovChain(self.Pl)
        mc.computePi('linear')
        self.eigen_stationarydist_Pl =  {'eigen values': np.around(w, 3), 'eigen vector': np.around(vr, 3),'stationary distribution': np.around(mc.pi,3)}
        return self.eigen_stationarydist_Pl

    def getv (self):
        return np.around(self.v, 3)
    
    def getu (self):
        return np.around(self.u, 3)

    def getPl (self):
        return np.around(self.Pl, 3)

    def getPl_normalize_stochastic (self):
        return np.around(self.Pl_normalize_stochastic,3)


