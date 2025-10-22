#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 18:17:02 2025

@author: EricPgh
"""

#block2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
bPlot=False
bPlotEig=False
L= 1.0 #mm 1e-3
Lunits = 'mm'
Nx=2048
Linert = 100
Rinert = 100
Mactive = Nx-Linert-Rinert
x = np.linspace(0.0, L, Nx)
xcenter = x[Linert:Mactive]
sides = ['left','right']
levels = [0.8,0.9,1.0,1.1,1.2]
dTS = {}
dNPOP = {}
dFLUX = {}
dBIDX = {}
#encoding = "_{side}{lev:.1f}"
for i,side in enumerate(sides):
  for j,lev in enumerate(levels):
    variant = f"_{side}{lev:.1f}"
    with open(f"bte_1d_flux{variant}.pkl", "rb") as f:
        ts, npop, flux = pickle.load(f) #next time I should have x, xcenter come through the pickle
    npop *= 1e-8 #Scaling to natural dimension
    Nunits = '[1e8]'
    flux *= 1e-6
    Funits = '[1e6]'
    ts *= 1e6
    Tunits = '[1e-6]'
    dTS[variant] = ts[:]
    dNPOP[variant] = npop[:]
    dFLUX[variant] = flux[:]

from numpy.linalg import svd,eig
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

def loadGinterpolants(ts, field, nevery, deg_x=12, bPlot=False):
    G = [] #This is a list of Legendre coefficients every n'th time point
    time_slice = np.arange(0,len(ts),nevery,dtype=int)
    #time_slice = np.linspace(1,len(ts),20,dtype=int)
    Fshort = field[time_slice]
    for Fs in Fshort:
        c=decompose_field(xcenter,Fs.T[Linert:Mactive], deg_x)
        G.append(c[:,0]) #chatgpt says unnecessary [:,0])
        if bPlot: #Plot efficiency of Legendre interpolation
            F = interp_temp(xcenter,c)
            plt.plot(xcenter ,F)
            plt.plot(xcenter ,Fs.T[Linert:Mactive],'o',markevery=100)
    if bPlot: #Plot efficiency of Legendre interpolation
        plt.xlim(0,L)
        plt.xlabel(f"x [{Lunits}]")
        plt.ylabel(f"{Lunits}")
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

deg_Leg = 12
dGN = {}
dGF = {}
for i,side in enumerate(sides):
    for j,lev in enumerate(levels):
        variant = f"_{side}{lev:.1f}"
        dGN[variant] = loadGinterpolants(ts, dNPOP[variant], 1, deg_Leg)
        dGF[variant] = loadGinterpolants(ts, dFLUX[variant], 1, deg_Leg)

#from linear_interp import Propagator
from cubic_interp import Propagator
prop = Propagator()

import math

def nDMD(n):
    return 2 + round(28 * (1 - math.tanh(0.015 * n)))
def cDMD(n):
    return 1 + round(29 * (1 - math.tanh(0.015 * n)))

for j,major_stride in enumerate([16]): #Just running with one for now but available for testing
    prop.major_stride = major_stride
    nA = 13000//major_stride #The quantity of propagators to compute
    lAc = [] #The running list of each propagator, quantity nA
    lTimes = []
    strt = 1
    minor_stride = 1 #not skipping timesteps within prior/posterior observations
    for i in range(nA):
        lX = []
        lY = []
        for j,side in enumerate(sides):
            for k,lev in enumerate(levels):
                variant = f"_{side}{lev:.1f}"
                idx = j  #which index of the boundary flux vector should get assigned the level
                nidx = len(sides)#*ndeg_bv #what is the dimension of the boundary flux vector, only operating with 1 degree (constant)

                npts = nDMD(i*major_stride)
                cmp = cDMD(i*major_stride)
                #Again using a second order scheme for integrating
                N0 = np.array(dGN[variant][0][strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
                F0 = np.array(dGF[variant][0][strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
                B0 = np.zeros( (nidx,npts) );B0[idx,:] = lev
                #print(N0.shape,F0.shape)
                #The priors X, are a composite of two coupled observations [N0;F0], both contributing to the prediction of Y, hence the A matrix has twice the size containing how Y evolves from each group of prior observations
                #Adding to this are the boundary flux levels which should impact the F posteriors
                lX.append( np.concatenate((N0,F0,B0),axis=0) )
                #print(i,X.shape)
                #Here Y is defined as the rate of change of observations. Alternatively could use just an observation of the system, but the physics being studied are nonlinear second order ODE (or nonlinear system of two first order ODE)
                N1 = np.array(dGN[variant][0][strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
                F1 = np.array(dGF[variant][0][strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
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
        lTimes.append(ts[strt+i*major_stride])#/2+ts[strt+(i+1)*major_stride]/2) #here I'm sampling times at the same rate (major_stride) such that I have time positions aligned with Ac transforms
        #This might not be occuring the way I imagine it. Maybe I should just get the delta_t and increment that in the integration loop
    #print(lTimes[0])
    prop.buildMe(lTimes,lAc)

#plot_A_elements(props,[ ([0.,0.05],'^'),
#                        ([0.05,0.2],'o'),
#                        ([0.2,min([props[j].t_list[-1] for j in range(2)])],'+') ])

import numpy as np
from scipy.integrate import solve_ivp

with open(f"bte_1d_flux.pkl", "rb") as f:
    ts, npop, flux = pickle.load(f) #next time I should have x, xcenter come through the pickle
npop *= 1e-8 #Scaling to natural dimension
Nunits = '[1e8]'
flux *= 1e-6
Funits = '[1e6]'
ts *= 1e6
Tunits = '[1e-6]'
bflux = np.array([1.,3.])

#With propagators formed, here begins the forward euler loop. this is intended to operate between large strides in the pickle solutions, starting with the beginning conditions of one stride and matching the final solutions (from BTE) at the end of the stride
Nstart=npop[0].T[Linert:Mactive] #population profile from the start
Fstart=flux[0].T[Linert:Mactive] #flux profile from the start
n0=decompose_field(xcenter,Nstart,deg_Leg) #Decompose the inside layer only, scale the temperature to natural units
f0=decompose_field(xcenter,Fstart,deg_Leg) #Decompose the inside layer only, scale the temperature to natural units
print(n0.shape,f0.shape)
c0=np.concatenate((n0,f0),axis=0).reshape(-1)  #(26,) #the boundary fluxes need to appear somehow (but RHS shouldn't have a derivative of it)
print(c0.shape)
nflux=np.concatenate((npop,flux),axis=0)
Nn = n0.shape[0]
#print(c0)
#Tappx = interp_temp(xcenter,c0)
#print(Tappx)
#plt.plot(xcenter ,Tappx)
#print(c)
nepoch=10 #I'm defining each epoch to be the period I'm running parallel integration alongside BTE solutions. Comparison happens at the epoch end. 
time_slice =np.linspace(100,13000,nepoch,dtype=int) #these i values are the epoch end points, so the first epoch runs 
#time_slice = time_slice[0:3]
#from i=0 to i=100, the next i=100 to i=590
#Notably I reused the list time_slice here. Previously it was returned by the loadGinterpolants(1) call and had a facile step of 1, here its being reused to define the epoch stepping

#I expect there to be some variation of forward euler with actual BTE. I'm defining here, with chatGPT guidance, a measurement operator H that is responsible for computing the measurement (Tavg) that can be compared to the BTE epoch measurement
avg_weights = [1.]
for _ in range(deg_Leg//2):
    avg_weights += [0.,1.]
if deg_Leg%2 != 0:
    avg_weights += [0.]
H = np.array(avg_weights,dtype=float).reshape((deg_Leg+1,1)) #Average of Legendre polynomial is the sum of even Pn coefficients, c0+c2+c4...

#This is maybe a less useful attempt. If I know what Tavg is at the end of each epoch by BTE, and the integration Tavg (by opeator H) is off by some difference, that difference (an integration quantity) needs to be distributed back to the Legendre coefficients before initiating the next epoch. This will ensure that Tavg matches between BTE and forward euler (FE) between epochs. This doesn't ensure the T profile will match. In some way should the H operator pseudoinversion matter? Or the current magnitudes of the Legendre coefficients be the weights for redistribution? The incorporation or implementation of the integration corrector is yet undefined and deserves consideration.
n = 2
spread = H/np.array([1,1,2,1,3,1,4,1,5,1,6,1,7],dtype=float).reshape((deg_Leg+1,1))**n
Hp = spread/sum(spread)
#Hp = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=float).reshape((deg_Leg+1,1))
#print(Hp)

#Begin integration
i=0
Navg_0 = 1. #Tavg at start of epoch, 0.1 is just this dataset and needs updated
t_0 = 0.
#print('ts',ts[time_slice])
colormap = plt.get_cmap('tab10', nepoch)
colors = [colormap(k) for k in range(nepoch)] #when plotting exact vs approx curves, each epoch has own color from cmap
for t, Cs, col in zip(ts[time_slice],nflux[time_slice],colors):#[0:1]: 10 elements to loop, epochs
    Navg = np.dot(H.T,decompose_field(xcenter,Cs[:Nn].T[Linert:Mactive])) #Ts is the BTE solution and this operation with H measures the average T at the epoch start
    #print(rhs(t_0,c0))
    #print(rhs(t,c0))
    if True:
        sol = solve_ivp(prop.rhs, (t_0, t), c0, args=( bflux ), method='RK45', atol=1e-8, rtol=1e-6, max_step= (t-t_0)/10.)
        #print('t',sol.t)
        #print('y',sol.y[:, -1])
        c1 = sol.y[:, -1]
        if False:
            for j in range(sol.y.shape[1]):
                plt.plot(xcenter ,interp_temp(xcenter,sol.y[:Nn,j]),color=col)
        else:
            plt.plot(xcenter ,interp_temp(xcenter,sol.y[:Nn,-1]),color=col)
        c0=c1 #update c_i solutions
    else:
      t_i = t_0 #(t-t_0)/10.+t_0
      dt = (t-t_0)/10.
      while t_i<t: #lTimes[i]<t: #lTimes was recorded during formation of the propagators, this is the within-epoch integration loop that runs until the time value of the propagator reaches t, the epoch end
        '''if i<7: #debugging
            print(lAc[i])
        else:
            break'''
        #The process is as such, the prior solution n0 and f0 are stacked and multiplied with current Ac (i'th) to provide dc(:=c1-c0)
        #dc = prop.rhs(lTimes[i],c0) #lAc[i]@c0
        dc = prop.rhs(t_i,c0)
        #however this difference is normalized by the minor stride interval (first deriv appx), but we are counting by major strides, so the difference, dc, needs to be scaled by major_stride to predict the increase of c2 over c1 when major_stride had elapsed
        #This won't run correctly, it works when multiplied by the stride size in index space
        #dc and rhs need to return in time space or be normalized by time step if SVD is impaired
        #If multiplying by stride integer, dc needs to be time agnostic, if multiplying by dt, dc needs to be normalized by timestep (applied in rhs())
        c1=c0+ dc*dt #major_stride*dc#*1.06 #e.g. +12*dc
        #print(c2)
        if False: #initial attempt to implement corrector. not sure why I was trying to do it within the epoch when I think by design I won't know Tavg except at epoch boundaries. I was having initial difficulties that I was attributing to integration errors and falsely though correctors were needed already. Final working prototype left the final line (c2+=Hp*(y-Hf)) commented and unused. but this workflow essential shows the chatGPT suggested process of distributing the error back into the coefficients by some operator Hp.
            Hf = np.dot(H.T,c1[Nn,:])
            norm = np.sqrt(np.dot(c1.T,c1)[0,0])
            #print(norm)
            y = (Navg-Navg_0)*(lTimes[i]-t_0)/(t-t_0)+Navg_0
            #print(Hf,y,Hp.shape,c2.shape)
            if False and i%10==0:
                print(c1.T)
            #print(y,Hf,Hp*(y-Hf))
            '''if i==1:
                break'''
            Hp = c1/norm #This seems reasonable in that the magnitude of the current solution vector elements should guide the distribution of the updates. Reasonable as first cut but not rigorous or empirically validated.
            #print(Hp)
            c1+=Hp*(y-Hf)
            #ChatGPT suggests this method under the idea that the vector update should be minimized.
            '''least-squares redistribution of measurement error:

            # Want minimal norm Δc so that H @ (c + Δc) = y_target
            # Δc = H^+ (y_target - H @ c)
            H_vec = H.ravel()   # shape (n_coeffs,)
            residual = (y_target - H_vec.dot(c))
            # pseudo-inverse for H: H^+ = H / (H.H) if H is vector
            Hp = H_vec / (H_vec.dot(H_vec) + 1e-12)
            delta_c = Hp * residual
            c_new = c + delta_c
            I do think this is too simplistic. It may develop oscillitory modes about the average.
            Other suggestions are minimizations of the update with hand tweaking. Another is Mahalanobis-style:
            Weighted (Mahalanobis) minimal-norm correction
            
            If you have prior knowledge of coefficient variances (covariance C), use
            \min \Delta c^T C^{-1} \Delta c\quad\text{s.t. } H\Delta c = r,
            leading to
            \Delta c = C H^T (H C H^T)^{-1} r.
            This will distribute correction across coefficients according to the covariance -- useful if some coefficients are known to be more trustworthy or you want to penalize changing high-frequency coefficients more strongly. An example snippet:
            import numpy as np
            
            # Suppose you have prior coefficient realizations: shape (N, n_c)
            C_samples = np.array([...])  # each row is c^(k)
            
            # Sample covariance
            C = np.cov(C_samples, rowvar=False)
            
            # Regularization with Ledoit–Wolf style that uses lambda to blend diagonal and off-diagonal to improve C condition.
            lam = 1e-6 * np.trace(C) / C.shape[0]
            C_reg = (1 - 0.1) * C + 0.1 * np.diag(np.diag(C)) + lam * np.eye(C.shape[0])
            
            # Then compute correction
            H = ...  # your measurement operator, shape (m, n_c)
            r = ...  # residual vector (y_target - H @ c_pred)
            
            Delta_c = C_reg @ H.T @ np.linalg.inv(H @ C_reg @ H.T) @ r'''
            
        c0=c1 #update c_i solutions
        if False and i%100==0:
            #print(c2.T)
            print(np.dot(H.T,c1[Nn,:]),y)
        #print(Ac)
        #print(c)
        i+=1
      plt.plot(xcenter ,interp_temp(xcenter,c1[:Nn]),color=col)
    plt.plot(xcenter ,Cs[:Nn].T[Linert:Mactive],'o',color=col,markevery=100)
    Navg_0 = Navg #Also part of an in-process corrector mechanism.
    t_0 = t
    #break
plt.xlim(0.,L)
plt.ylim(0.,1.01)
plt.xlabel(f"x [{Lunits}]")
plt.ylabel(f"{Nunits}")
plt.savefig('bte_predictor_performance.png')
plt.show()

