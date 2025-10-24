#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: EricPgh
"""

#block2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.linalg import svd,eig
import math
from cubic_interp import Propagator
#from linear_interp import Propagator
from scipy.integrate import solve_ivp


def decompose_field(x,T, deg_x= 12):
    from numpy.polynomial.legendre import legvander
    # Rescale coordinates to [-1, 1]
    x_scaled = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    
    # Get Vandermonde matrix (evaluated basis)
    V = legvander(x_scaled, deg_x)  # shape: (N_points)
    
    #The following least-squares call performs this task:
    # Project via inner product (or least-squares)
    # Result is vector of length (deg_x + 1)*(deg_y + 1)
    # Project T_flat onto the Legendre basis
    #G = V.T @ V           # Gram matrix
    #b = V.T @ T           # Unnormalized projection
    #c = np.linalg.solve(G, b)  # Solve for coefficients
    c, *_ = np.linalg.lstsq(V, T, rcond=1e-12)
    return c  # shape (deg_x+1,)
def interp_temp(x,c):
    from numpy.polynomial.legendre import legval
    x_scaled = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    return legval(x_scaled, c,tensor=False) #np.sin(np.pi * x / Lx)  # W/m²
#deprecated scale = lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1

def loadGinterpolants(learner,ts, field, nevery, deg_x=12, bPlot=False):
    G = [] #This is a list of Legendre coefficients every n'th time point
    time_slice = np.arange(0,len(ts),nevery,dtype=int)
    #time_slice = np.linspace(1,len(ts),20,dtype=int)
    Fshort = field[time_slice]
    for Fs in Fshort:
        c=decompose_field(learner.xcenter,Fs.T[learner.Linert:learner.Mactive], deg_x)
        G.append(c[:,0]) #chatgpt says unnecessary [:,0])
        if bPlot: #Plot efficiency of Legendre interpolation
            F = interp_temp(learner.xcenter,c)
            plt.plot(learner.xcenter ,F)
            plt.plot(learner.xcenter ,Fs.T[learner.Linert:learner.Mactive],'o',markevery=100)
    if bPlot: #Plot efficiency of Legendre interpolation
        plt.xlim(0,learner.L)
        plt.xlabel(f"x [{learner.Lunits}]")
        plt.ylabel(f"{learner.Lunits}")
        plt.savefig('bte_temps_interp.png')
        plt.show()
    return [G,time_slice]

def pseudo_inverse_from_svd(u, s, vt, r=None, tol=1e-12):
    if r is None:
        # choose rank by tol relative to max singular
        r = np.sum(s > tol * s.max())
    s_inv = np.zeros_like(s)
    s_inv[:r] = 1.0 / s[:r]
    #U_r = u[:, :r], S_r = s[:r], Vt_r = v[:r, :] (depending on numpy.linalg.svd output u,s,vt).
    #v.T[:,0:cmp]@np.diag(sinv[0:cmp])@u.T[0:cmp,:]
    return (vt.T[:, :r] @ np.diag(s_inv[:r])) @ u.T[:r, :]  # returns X^+ (shape matches)





class HeatLearner:
    def __init__(self):
        self.bPlot=False
        self.bPlotEig=False
        self.L= 1.0 #mm 1e-3
        self.Lunits = 'mm'
        self.Nx=2048
        self.Linert = 100
        self.Rinert = 100
        self.Mactive = self.Nx-self.Linert-self.Rinert
        self.x = np.linspace(0.0, self.L, self.Nx)
        self.xcenter = self.x[self.Linert:self.Mactive]
        self.sides = ['left','right']
        self.levels = [0.8,0.9,1.0,1.1,1.2]
        self.dTS = {}
        self.dNPOP = {}
        self.dFLUX = {}
        #encoding = "_{side}{lev:.1f}"
        for i,side in enumerate(self.sides):
          for j,lev in enumerate(self.levels):
            variant = f"_{side}{lev:.1f}"
            with open(f"bte_1d_flux{variant}.pkl", "rb") as f:
                ts, npop, flux = pickle.load(f) #next time I should have x, xcenter come through the pickle
            npop *= 1e-8 #Scaling to natural dimension
            self.Nunits = '[1e8]'
            flux *= 1e-6
            self.Funits = '[1e6]'
            ts *= 1e6
            self.Tunits = '[1e-6]'
            self.dTS[variant] = ts[:]
            self.dNPOP[variant] = npop[:]
            self.dFLUX[variant] = flux[:]
        self.deg_Leg = 12
        self.dGN = {}
        self.dGF = {}
        self.prop = Propagator()

    def digest_fields(self):
        for i,side in enumerate(self.sides):
            for j,lev in enumerate(self.levels):
                variant = f"_{side}{lev:.1f}"
                self.dGN[variant] = loadGinterpolants(self,self.dTS[variant], self.dNPOP[variant], 1, self.deg_Leg)
                self.dGF[variant] = loadGinterpolants(self,self.dTS[variant], self.dFLUX[variant], 1, self.deg_Leg)

