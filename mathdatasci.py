import numpy as np
from scipy import linalg
from discreteMarkovChain import markovChain



class mathdatasci:
    def __init__(self, P, l):
        self.P = P
        self.l = l
        self.v = np.ndarray((2,2))
        self.u = np.ndarray((2,2))
        self.Pl = np.empty(0)
        self.Pl_normalize_stochastic = np.empty(0)
        self.eigen_stationarydist = {}
        self.get_v()
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
        Pl_normalize_stochastic = self.Pl
        for i in range (int (np.sqrt(Pl_normalize_stochastic.size))):
            k = 1 - Pl_normalize_stochastic[i][i]
            Pl_normalize_stochastic[i][i] = 0
            Pl_normalize_stochastic[i]  =Pl_normalize_stochastic[i]* ((k)**(-1))
        self.Pl_normalize_stochastic = Pl_normalize_stochastic

    def get_eigen_stationarydist(self):
        w, vl, vr = linalg.eig(self.Pl_normalize_stochastic, left=True)
        mc = markovChain(self.Pl_normalize_stochastic)
        mc.computePi('linear')
        self.eigen_stationarydist =  {'eigen values': w, 'eigen vector': vl,'stationary distribution': mc.pi}
        return self.eigen_stationarydist
