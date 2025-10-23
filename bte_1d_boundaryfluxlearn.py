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
from scipy.integrate import solve_ivp
from HeatLearner import decompose_field, interp_temp

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
    #time_slice = time_slice[0:3]
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
        if True:
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
          while t_i<t: #lTimes[i]<t: #lTimes was recorded during formation of the propagators, this is the within-epoch integration loop that runs until the time value of the propagator reaches t, the epoch end
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
    
