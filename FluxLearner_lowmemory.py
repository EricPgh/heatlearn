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
        c=decompose_field(learner.x,Fs.T, deg_x)
        #c=decompose_field(learner.xcenter,Fs.T[learner.Linert:learner.Mactive], deg_x)
        G.append(c[:,0]) #chatgpt says unnecessary [:,0])
        if bPlot: #Plot efficiency of Legendre interpolation
            F = interp_temp(learner.x,c)
            plt.plot(learner.x ,F)
            #plt.plot(learner.x ,Fs.T[learner.Linert:learner.Mactive],'o',markevery=100)
            plt.plot(learner.x ,Fs.T,'o',markevery=100)
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





class FluxLearner:
    def __init__(self):
        pp='pkls/'
        self.bPlot=False
        self.bPlotEig=False
        self.L= 1.0 #mm 1e-3
        self.Lunits = 'mm'
        self.mflux = 1e0 #-9
        self.mts = 1e10 #1e6
        self.mnpop = 1e0 #Scaling to natural dimension
        self.Nunits = '[1e0]'
        self.Nx=2048
        self.Linert = 100
        self.Rinert = 100
        self.Mactive = self.Nx-self.Linert-self.Rinert
        self.x = np.linspace(0.0, self.L, self.Nx)
        self.xcenter = self.x[self.Linert:self.Mactive]
        self.sides = ['left','right']
        self.levels = [0.8,0.9,1.0,1.1,1.2]
        self.ndeg = 25
        self.TS = None
        #self.dFLUX = [{} for _ in range(self.ndeg+1)]
        #self.dNPOP = [{} for _ in range(self.ndeg+1)]
        self.dCF =  [{} for _ in range(self.ndeg+1)]
        self.dCN =  [{} for _ in range(self.ndeg+1)]
        self.deg_Leg = 24
        #encoding = "_{side}{lev:.1f}"
        for variant in ["_ramp_rxn1.0",]+[f"_ramp_rxn{load:.1f}" for load in np.arange(5.,101.,5.)]+[f"_ramp",f"_center"]:
            with open(f"{pp}bte_1d_flux{variant}.pkl", "rb") as f:
                ts, npop, flux = pickle.load(f) #next time I should have x, xcenter come through the pickle
            npop *= self.mnpop #Scaling to natural dimension
            self.Nunits = '[1e0]'
            flux *= self.mflux
            self.Funits = '[1e0]'
            ts *= self.mts
            self.Tunits = '[1e-10]' #'[1e-6]'
            self.TS = ts[:]
            self.dCF[self.ndeg][variant] = loadGinterpolants(self,ts, flux, 1, self.deg_Leg)
            self.dCN[self.ndeg][variant] = loadGinterpolants(self,ts, npop, 1, self.deg_Leg)
            #self.dFLUX[self.ndeg][variant] = flux[:]
        for deg in range(self.ndeg):
            for i,side in enumerate(self.sides):
                for j,lev in enumerate(self.levels):
                    variant = f"_{side}{lev:.1f}"
                    with open(f"{pp}bte_1d_flux_deg{deg}{variant}.pkl", "rb") as f:
                        ts, npop, flux = pickle.load(f) #next time I should have x, xcenter come through the pickle
                    npop *= self.mnpop #Scaling to natural dimension
                    self.Nunits = '[1e0]'
                    flux *= self.mflux
                    self.Funits = '[1e0]'
                    ts *= self.mts
                    self.Tunits = '[1e-10]' #'[1e-6]'
                    self.TS = ts[:] #I know I ought to have room for variable timesteps but its complicating
                    self.dCF[deg][variant] = loadGinterpolants(self,ts, flux, 1, self.deg_Leg)
                    #self.dFLUX[deg][variant] = flux[:]
        self.Ginf = None
        self.prop = Propagator()
        self.gf = Propagator()

    def digest_boundary_fields(self):
        for deg in range(self.ndeg):
            for i,side in enumerate(self.sides):
                for j,lev in enumerate(self.levels):
                    variant = f"_{side}{lev:.1f}"
                    self.dCF[deg][variant] = loadGinterpolants(self,self.dTS[deg][variant], self.dFLUX[deg][variant], 1, self.deg_Leg)

    def digest_fields(self):
        variant = f"_rxn20.0"
        deg = self.ndeg
        self.dCF[deg][variant] = loadGinterpolants(self,self.dTS[deg][variant], self.dFLUX[deg][variant], 1, self.deg_Leg)
        self.dCN[deg][variant] = loadGinterpolants(self,self.dTS[deg][variant], self.dNPOP[deg][variant], 1, self.deg_Leg)
