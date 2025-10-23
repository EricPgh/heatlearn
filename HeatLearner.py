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

def plot_A_elements(props,ts_m, n_points=200):
    """Plot all A_ij(t) trajectories from the Propagator instance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for ts,m in ts_m:
        t_dense = np.linspace(ts[0], ts[1], n_points)
        n0, n1 = props[0].dim0, props[0].dim1

        colormap = plt.get_cmap('tab10', n0*n1)
        colors = [colormap(k) for k in range(n0*n1)] #when plotting exact vs approx curves, each epoch has own color from cmap
        for i in range(n0):
            for j in range(n1):
                col = colors[i*n1+j]
                A_vals = [ [props[k].splines[i][j](t) for t in t_dense] for k in range(2)]
                #ax.plot(t_dense, A_vals, label=f"A[{i},{j}]")
                ax.plot(A_vals[0], A_vals[1], linestyle='',marker=m,color=col)

        ax.set_xlabel("Time")
        ax.set_ylabel("A(t) elements")
        ax.set_title(f"Time evolution of A(t) matrix elements s+=, npts,cmp variable")
    #ax.legend(fontsize=8, ncol=3, loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    #plt.xlim(-.0004,.0005)
    #plt.ylim(-.0004,.0005)
    plt.show()




def nDMD(n):
    return 2 + round(28 * (1 - math.tanh(0.015 * n)))
def cDMD(n):
    return 1 + round(29 * (1 - math.tanh(0.015 * n)))

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

    def buildPropagators(self):
        for j,major_stride in enumerate([16]): #Just running with one for now but available for testing
            self.prop.major_stride = major_stride
            nA = 13000//major_stride #The quantity of propagators to compute
            lAc = [] #The running list of each propagator, quantity nA
            lTimes = []
            strt = 1
            minor_stride = 1 #not skipping timesteps within prior/posterior observations
            for i in range(nA):
                lX = []
                lY = []
                for j,side in enumerate(self.sides):
                    for k,lev in enumerate(self.levels):
                        variant = f"_{side}{lev:.1f}"
                        idx = j  #which index of the boundary flux vector should get assigned the level
                        nidx = len(self.sides)#*ndeg_bv #what is the dimension of the boundary flux vector, only operating with 1 degree (constant)
        
                        npts = nDMD(i*major_stride)
                        cmp = cDMD(i*major_stride)
                        #Again using a second order scheme for integrating
                        N0 = np.array(self.dGN[variant][0][strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
                        F0 = np.array(self.dGF[variant][0][strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
                        B0 = np.zeros( (nidx,npts) );B0[idx,:] = lev
                        #print(N0.shape,F0.shape)
                        #The priors X, are a composite of two coupled observations [N0;F0], both contributing to the prediction of Y, hence the A matrix has twice the size containing how Y evolves from each group of prior observations
                        #Adding to this are the boundary flux levels which should impact the F posteriors
                        lX.append( np.concatenate((N0,F0,B0),axis=0) )
                        #print(i,X.shape)
                        #Here Y is defined as the rate of change of observations. Alternatively could use just an observation of the system, but the physics being studied are nonlinear second order ODE (or nonlinear system of two first order ODE)
                        N1 = np.array(self.dGN[variant][0][strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
                        F1 = np.array(self.dGF[variant][0][strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
                        dN = (N1-N0)/minor_stride
                        dF = (F1-F0)/minor_stride
                        #The boundary levels don't appear in the Y values because they are priors only, not posteriors
                        lY.append( np.concatenate((dN,dF),axis=0) ) #The posteriors Y, are a composite of two coupled difference observations [dN/dt;dF/dt], e.g. the state of Y, hence the A matrix has 4x4 block form containing how Y evolves from each group of prior observations
                #It can be seen that the Y observation is X1-X0 while the X observation is X0. Thus the system behavior being trained is X1-X0 = A(X0). Given X0, the A propagator will yield X1-X0 and then X0. This Y observation is divided by the minor stride timestep to yield a proper first derivative approximate
                X = np.concatenate(lX,axis=1)
                u,s,vt = svd(X,full_matrices=False) #full matrices=false, want to be able to compress, S has nonzero only
                s += 0.001
                #print(s)
                #A = Y@v.T@np.diag(sinv)@u.T
                X_pinv = pseudo_inverse_from_svd(u, s, vt, cmp) # v.T[:,0:cmp]@np.diag(sinv[0:cmp])@u.T[0:cmp,:]
                Y = np.concatenate(lY,axis=1)
                Ac = Y@X_pinv #taking the 50% compression, this also seems to stabilise the approximation, maybe consider a study on the variation of this with integration accuracy
                lAc.append(Ac) #this will contain a time evolution of differential propagators
                #lTimes.append(ts[strt+i*major_stride+minor_stride]/2+ts[strt+i*major_stride+npts*minor_stride]/2) #here I'm sampling times at the same rate (major_stride) such that I have time positions aligned with Ac transforms
                lTimes.append(self.dTS[variant][strt+i*major_stride])#/2+ts[strt+(i+1)*major_stride]/2) #here I'm sampling times at the same rate (major_stride) such that I have time positions aligned with Ac transforms
                #This might not be occuring the way I imagine it. Maybe I should just get the delta_t and increment that in the integration loop
            #print(lTimes[0])
            self.prop.buildMe(lTimes,lAc)
        
        #plot_A_elements(props,[ ([0.,0.05],'^'),
        #                        ([0.05,0.2],'o'),
        #                        ([0.2,min([props[j].t_list[-1] for j in range(2)])],'+') ])

