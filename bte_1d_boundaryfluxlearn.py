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
import math
from scipy.integrate import solve_ivp
from numpy.linalg import svd,eig
from HeatLearner import decompose_field, interp_temp, pseudo_inverse_from_svd

def nDMD(n):
    return 2 + round(28 * (1 - math.tanh(0.015 * n)))
def cDMD(n):
    return 1 + round(29 * (1 - math.tanh(0.015 * n)))


def plot_A_elements(prop,ts_m, n_points=200):
    """Plot all A_ij(t) trajectories from the Propagator instance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for ts,m in ts_m:
        t_dense = np.linspace(ts[0], ts[1], n_points)
        n0, n1 = prop.dim0, prop.dim1

        colormap = plt.get_cmap('tab10', n0*n1)
        colors = [colormap(k) for k in range(n0*n1)] #when plotting exact vs approx curves, each epoch has own color from cmap
        for i in range(n0):
            for j in range(n1):
                col = colors[i*n1+j]
                A_vals = [prop.splines[i][j](t) for t in t_dense]
                ax.plot(t_dense, A_vals, linestyle='',marker=m,color=col) #, label=f"A[{i},{j}]")

        ax.set_xlabel("Time")
        ax.set_ylabel("A(t) elements")
        ax.set_title(f"Time evolution of A(t) matrix elements s+=, npts,cmp variable")
    #ax.legend(fontsize=8, ncol=3, loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.xlim(0.,.05)
    plt.ylim(-.04,.05)
    plt.show()

def plot_A_element_comparison(props,ts_m, n_points=200):
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

def buildPropagators(HL):
    for j,major_stride in enumerate([16]): #Just running with one for now but available for testing
        HL.prop.major_stride = major_stride
        nA = 13000//major_stride #The quantity of propagators to compute
        lAc = [] #The running list of each propagator, quantity nA
        lTimes = []
        strt = 1
        minor_stride = 1 #not skipping timesteps within prior/posterior observations
        for i in range(nA):
            lX = []
            lY = []
            for j,side in enumerate(HL.sides):
                for k,lev in enumerate(HL.levels):
                    variant = f"_{side}{lev:.1f}"
                    idx = j  #which index of the boundary flux vector should get assigned the level
                    nidx = len(HL.sides)#*ndeg_bv #what is the dimension of the boundary flux vector, only operating with 1 degree (constant)
    
                    npts = nDMD(i*major_stride)
                    cmp = cDMD(i*major_stride)
                    #Again using a second order scheme for integrating
                    N0 = np.array(HL.dGN[variant][0][strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
                    F0 = np.array(HL.dGF[variant][0][strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
                    B0 = np.zeros( (nidx,npts) );B0[idx,:] = lev
                    #print(N0.shape,F0.shape)
                    #The priors X, are a composite of two coupled observations [N0;F0], both contributing to the prediction of Y, hence the A matrix has twice the size containing how Y evolves from each group of prior observations
                    #Adding to this are the boundary flux levels which should impact the F posteriors
                    lX.append( np.concatenate((N0,F0,B0),axis=0) )
                    #print(i,X.shape)
                    #Here Y is defined as the rate of change of observations. Alternatively could use just an observation of the system, but the physics being studied are nonlinear second order ODE (or nonlinear system of two first order ODE)
                    N1 = np.array(HL.dGN[variant][0][strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
                    F1 = np.array(HL.dGF[variant][0][strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
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
            lTimes.append(HL.dTS[variant][strt+i*major_stride])#/2+ts[strt+(i+1)*major_stride]/2) #here I'm sampling times at the same rate (major_stride) such that I have time positions aligned with Ac transforms
            #This might not be occuring the way I imagine it. Maybe I should just get the delta_t and increment that in the integration loop
        #print(lTimes[0])
        HL.prop.buildMe(lTimes,lAc)
    
    #plot_A_elements(props,[ ([0.,0.05],'^'),
    #                        ([0.05,0.2],'o'),
    #                        ([0.2,min([props[j].t_list[-1] for j in range(2)])],'+') ])


def integrate(HL):
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
    Nstart=npop[0].T[HL.Linert:HL.Mactive] #population profile from the start
    Fstart=flux[0].T[HL.Linert:HL.Mactive] #flux profile from the start
    n0=decompose_field(HL.xcenter,Nstart,HL.deg_Leg) #Decompose the inside layer only, scale the temperature to natural units
    f0=decompose_field(HL.xcenter,Fstart,HL.deg_Leg) #Decompose the inside layer only, scale the temperature to natural units
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
    time_slice = time_slice[0:3]
    #from i=0 to i=100, the next i=100 to i=590
    #Notably I reused the list time_slice here. Previously it was returned by the loadGinterpolants(1) call and had a facile step of 1, here its being reused to define the epoch stepping
    
    #I expect there to be some variation of forward euler with actual BTE. I'm defining here, with chatGPT guidance, a measurement operator H that is responsible for computing the measurement (Tavg) that can be compared to the BTE epoch measurement
    avg_weights = [1.]
    for _ in range(HL.deg_Leg//2):
        avg_weights += [0.,1.]
    if HL.deg_Leg%2 != 0:
        avg_weights += [0.]
    H = np.array(avg_weights,dtype=float).reshape((HL.deg_Leg+1,1)) #Average of Legendre polynomial is the sum of even Pn coefficients, c0+c2+c4...
    
    #This is maybe a less useful attempt. If I know what Tavg is at the end of each epoch by BTE, and the integration Tavg (by opeator H) is off by some difference, that difference (an integration quantity) needs to be distributed back to the Legendre coefficients before initiating the next epoch. This will ensure that Tavg matches between BTE and forward euler (FE) between epochs. This doesn't ensure the T profile will match. In some way should the H operator pseudoinversion matter? Or the current magnitudes of the Legendre coefficients be the weights for redistribution? The incorporation or implementation of the integration corrector is yet undefined and deserves consideration.
    n = 2
    spread = H/np.array([1,1,2,1,3,1,4,1,5,1,6,1,7],dtype=float).reshape((HL.deg_Leg+1,1))**n
    Hp = spread/sum(spread)
    #Hp = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=float).reshape((HL.deg_Leg+1,1))
    #print(Hp)
    
    #Begin integration
    i=0
    #Navg_0 = 1. #Tavg at start of epoch, 0.1 is just this dataset and needs updated
    t_0 = 0.
    #print('ts',ts[time_slice])
    colormap = plt.get_cmap('tab10', nepoch)
    colors = [colormap(k) for k in range(nepoch)] #when plotting exact vs approx curves, each epoch has own color from cmap
    for t, Cs, col in zip(ts[time_slice],nflux[time_slice],colors):#[0:1]: 10 elements to loop, epochs
        #Navg = np.dot(H.T,decompose_field(HL.xcenter,Cs[:Nn].T[HL.Linert:HL.Mactive])) #Ts is the BTE solution and this operation with H measures the average T at the epoch start
        #print(rhs(t_0,c0))
        #print(rhs(t,c0))
        if False:
            sol = solve_ivp(HL.prop.rhs, (t_0, t), c0, args=( bflux ), method='RK45', atol=1e-8, rtol=1e-6, max_step= (t-t_0)/10.)
            #print('t',sol.t)
            #print('y',sol.y[:, -1])
            c1 = sol.y[:, -1]
            if False:
                for j in range(sol.y.shape[1]):
                    plt.plot(HL.xcenter ,interp_temp(HL.xcenter,sol.y[:Nn,j]),color=col)
            else:
                plt.plot(HL.xcenter ,interp_temp(HL.xcenter,sol.y[:Nn,-1]),color=col)
            c0=c1 #update c_i solutions
        else:
            t_i = t_0 #(t-t_0)/10.+t_0
            dt = (t-t_0)/10.
            while t_i<=t: #lTimes[i]<t: #lTimes was recorded during formation of the propagators, this is the within-epoch integration loop that runs until the time value of the propagator reaches t, the epoch end
                '''if i<7: #debugging
                    print(lAc[i])
                else:
                    break'''
                #The process is as such, the prior solution n0 and f0 are stacked and multiplied with current Ac (i'th) to provide dc(:=c1-c0)
                #dc = prop.rhs(lTimes[i],c0) #lAc[i]@c0
                dc = HL.prop.rhs(t_i,c0)
                #however this difference is normalized by the minor stride interval (first deriv appx), but we are counting by major strides, so the difference, dc, needs to be scaled by major_stride to predict the increase of c2 over c1 when major_stride had elapsed
                #This won't run correctly, it works when multiplied by the stride size in index space
                #dc and rhs need to return in time space or be normalized by time step if SVD is impaired
                #If multiplying by stride integer, dc needs to be time agnostic, if multiplying by dt, dc needs to be normalized by timestep (applied in rhs())
                c1=c0+ dc*dt #major_stride*dc#*1.06 #e.g. +12*dc
                #print(c2)
                c0=c1 #update c_i solutions
                i+=1
                if t-t_i>=dt:
                    t_i+=dt
                else:
                    t_i = t
                plt.plot(HL.xcenter ,interp_temp(HL.xcenter,c1[:Nn]),color=col)
        plt.plot(HL.xcenter ,Cs[:Nn].T[HL.Linert:HL.Mactive],'o',color=col,markevery=100)
        #Navg_0 = Navg #Also part of an in-process corrector mechanism.
        t_0 = t
        #break
    plt.xlim(0.,HL.L)
    plt.ylim(0.,1.01)
    plt.xlabel(f"x [{HL.Lunits}]")
    plt.ylabel(f"{HL.Nunits}")
    plt.savefig('bte_boundary_predictor_performance.png')
    plt.show()
    
