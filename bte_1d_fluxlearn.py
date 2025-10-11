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
bPlot=True
bPlotEig=False
L= 1.0 #mm 1e-3
Lunits = 'mm'
Nx=2048
Linert = 100
Rinert = 100
Mactive = Nx-Linert-Rinert
x = np.linspace(0.0, L, Nx)
xcenter = x[Linert:Mactive]
with open("bte_1d_flux.pkl", "rb") as f:
    ts, npop, flux = pickle.load(f) #next time I should have x, xcenter come through the pickle
npop *= 1e-9 #Scaling to natural dimension
Nunits = '[1e9]'
flux *= 1e-6
Funits = '[1e6]'
if bPlot:# Just a little plotting of as-loaded contours
    mpl.rcParams.update({"figure.figsize": (7, 4)})
    time_slice = np.linspace(1,len(ts),20,dtype=int)
    ts_short = ts[time_slice]
    Nshort = npop[time_slice]
    for Ns in Nshort:
    #for Ts in Tsnaps:#[np.array([1,6,-1])]:
        plt.plot(x , Ns.T)#, label="?(x, t_final)")
    plt.xlabel(f"x [{Lunits}]")
    plt.ylabel(f"{Nunits}")
    plt.title("1D Inventory from Upwind BTE")
    #plt.legend()
    plt.tight_layout()
    plt.xlim(0,L)
    plt.ylim(0,1.)
    plt.savefig('bte_temps_as-is.png')
    plt.show()

#block3

from numpy.linalg import svd,eig
fig, axs = plt.subplots(figsize=(6,4))
def decompose_field(x,T, deg_x= 12):
    from numpy.polynomial.legendre import legvander,legval
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

def loadGinterpolants(ts, field, nevery, bPlot=False):
    G = [] #This is a list of Legendre coefficients every n'th time point
    time_slice = np.arange(0,len(ts)+1,nevery,dtype=int)
    #time_slice = np.linspace(1,len(ts),20,dtype=int)
    ts_short = ts[time_slice]
    Fshort = field[time_slice]
    for Fs in Fshort:
        c=decompose_field(xcenter,Fs.T[Linert:Mactive])
        G.append(c) #unnecessary [:,0])
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
    return G,time_slice

def pseudo_inverse_from_svd(u, s, vt, r=None, tol=1e-12):
    if r is None:
        # choose rank by tol relative to max singular
        r = np.sum(s > tol * s.max())
    s_inv = np.zeros_like(s)
    s_inv[:r] = 1.0 / s[:r]
    #U_r = u[:, :r], S_r = s[:r], Vt_r = v[:r, :] (depending on numpy.linalg.svd output u,s,vt).
    #v.T[:,0:cmp]@np.diag(sinv[0:cmp])@u.T[0:cmp,:]
    return (vt.T[:, :r] * s_inv) @ u.T[:r, :]  # returns X^+ (shape matches)

G,time_slice = loadGinterpolants(ts, npop, 20)
ts_short=ts[time_slice]
#print(np.array(G))
#This next section is a first order attempt to visualize the system and eigenvalues, it has yet to be interesting
if bPlotEig:
    fig, axs = plt.subplots(9,5,figsize=(8,12))
Gt = [] #This contains slices of the G list, DMD/SVD uses blocks of consecutive observations
print(len(G))
cmp=11 #fidelity level (vice compression), how many singular values to keep
for i in range(len(G)-11):
    Gt.append(G[i:11+i]) #each element contains a block slice of G
    X = np.array(G[i:11+i]) #DMD considers the prior grouping of observations
    Y = np.array(G[i+1:12+i]) #DMD measures the system by its evolution to posterior observations
    #print(X.shape,Y.shape)
    #Presume Y(posterior)=A(system transform)*X(prior), then Y=AUSV
    u,s,vt = svd(X,full_matrices=False) #full matrices=false, want to be able to compress, S has nonzero only
    #print(s)
    #print(Y.shape,v.T[:,0:cmp].shape,u.T[0:cmp,:].shape)
    #sinv = 1./s #diagonal singular values, I guess vectorized
    X_pinv = pseudo_inverse_from_svd(u, s, vt) # v.T@np.diag(sinv)@u.T
    A = Y@X_pinv #Use SVD as pseudoinverse of X on Y to get system response
    X_pinv = pseudo_inverse_from_svd(u, s, vt, r=cmp) # v.T[:,0:cmp]@np.diag(sinv[0:cmp])@u.T[0:cmp,:]
    Ac = Y@X_pinv #compressed versions
    Lam,Xi = eig(A)
    Lam = np.diag(Lam)
    if bPlotEig: #Does the eigen-decomp ever show anything useful?
        #for ax,M in zip(axs[i,:],[u[:,0:cmp],np.diag(s[0:cmp]),v[0:cmp,:],A,Ac]):#,Lam,Xi.real,Xi.imag]):#[0,:]):
        for ax,M in zip(axs[i,:],[A,Ac,Lam,Xi.real,Xi.imag]):#[0,:]):
            im = ax.imshow(M, cmap='hot', vmin=-.2,vmax=.2,
                   origin='upper', interpolation='nearest', aspect='equal')
            ax.set_xlim(0,L)
            ax.set_ylim(0,1.)
        fig.savefig('bte_temps_eigs.png')
        fig.show()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #plt.title(f'Green\'s Function after {Nt} steps')
        #plt.xlabel(f"x [{Lunits}]")
        #plt.ylabel(f'{Nunits}')
        plt.tight_layout()
        #plt.savefig('matrix_evolution.png')
        plt.show()

#block4
GN,time_slice = loadGinterpolants(ts, npop, 1)
GF,time_slice = loadGinterpolants(ts, flux, 1)

#block5
#print(np.array(G))
nA = 6 #The number of axes I want to plot over
nM = 5 #The number of matrices in each row
fig, axs = plt.subplots(nA,nM,figsize=(nM,2*nA))
Gt = []
lAc = []
print(len(GN))
cmp=3 #fidelity or compression again, up to npts
#for i in range(0,len(G)-11,1):
strt = 1 #where in the total pickle file should the imaging start, beginning, middle, end?
major_stride = 12 #major is how many timepoints to skip for each axis for imaging
minor_stride = 1 #minor is how many timepoints between each column of the X0,X1 observations
npts = 6 #how many observations to include in each SVD
for i,ax in enumerate(axs):
    #Gt.append(G[i:11+i]) #each element contains a block slice of GN
    #The 2x first order version uses priors N0 and F0 to predict posterior Y, e.g. dN,dF
    N0 = np.array(GN[strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
    F0 = np.array(GF[strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T

    #print(N0.shape,F0.shape)
    X  = np.concatenate((N0,F0),axis=0) #The priors X, are a composite of two coupled observations [N0;F0], both contributing to the prediction of Y, hence the A matrix has twice the size containing how Y evolves from each group of prior observations
    #Here Y is defined as the rate of change of observations. Alternatively could use just an observation of the system, but the physics being studied are nonlinear second order ODE (or nonlinear system of two first order ODE)
    N1 = np.array(GN[strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
    F1 = np.array(GF[strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
    dN = (N1-N0)/minor_stride
    dF = (F1-F0)/minor_stride
    Y  = np.concatenate((dN,dF),axis=0) #The posteriors Y, are a composite of two coupled difference observations [dN/dt;dF/dt], e.g. the state of Y, hence the A matrix has 4x4 block form containing how Y evolves from each group of prior observations
    #print(X.shape,Y.shape)
    #Presume Y(posterior)=A(system transform)*X(prior), then Y=AUSV
    u,s,vt = svd(X,full_matrices=False) #full matrices=false, want to be able to compress, S has nonzero only
    #print(s)
    #print(u.shape,v.shape)
    x_deg = u.shape[0]//2
    #print(Y.shape,v.T[:,0:cmp].shape,u.T[0:cmp,:].shape)
    X_pinv = pseudo_inverse_from_svd(u, s, vt) # v.T@np.diag(sinv)@u.T
    A = Y@X_pinv #Use SVD as pseudoinverse of X on Y to get system response
    X_pinv = pseudo_inverse_from_svd(u, s, vt, r=cmp) # v.T[:,0:cmp]@np.diag(sinv[0:cmp])@u.T[0:cmp,:]
    Ac = Y@X_pinv #compressed versions
    #print(A[:,0:x_deg].shape)
    #I don't remember what this next comment block was doing
    '''Anorm1 = np.linalg.norm(A[:,0:x_deg],ord=2)
    norm_comm1 = np.linalg.norm(A[:,0:x_deg] @ A[:,0:x_deg].conj().T - A[:,0:x_deg].conj().T @ A[:,0:x_deg], ord=2)
    Anorm0 = np.linalg.norm(A[:,x_deg:2*x_deg],ord=2)
    norm_comm0 = np.linalg.norm(A[:,x_deg:2*x_deg] @ A[:,x_deg:2*x_deg].conj().T - A[:,x_deg:2*x_deg].conj().T @ A[:,x_deg:2*x_deg], ord=2)
    #print(Anorm1,norm_comm1,Anorm0,norm_comm0)
    
    u1 = u[0:x_deg,:]
    u0 = u[x_deg:2*x_deg,:]
    Ardc1 = u1.T@Y@v.T@np.diag(sinv)
    Ardc0 = u0.T@Y@v.T@np.diag(sinv)
    Lam,Xi = eig(Ardc0)
    Lam = np.diag(np.abs(Lam))'''
    lAc.append(A) #list of transformations from one major stride to next
    for axi,M in zip(ax,[u[:,:cmp],np.diag(s[:cmp]),v[:cmp,:],Ac,A,Lam,Xi.real,Xi.imag][:nM]):
        im = axi.imshow(M, cmap='hot', vmin=-1,vmax=1,
           origin='upper', interpolation='nearest', aspect='equal')
        fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)
#fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig('bte_2x1stOrd_transforms1.png')

plt.show()
fig, axs = plt.subplots(nA,1,figsize=(8,12))

print(len(lAc))
for ax,i in zip(axs, list(range(0,nA))):#len(lAc),100))[:9]):
    im = ax.imshow(lAc[i], cmap='hot', #vmin=-15,vmax=5,
           origin='upper', interpolation='nearest', aspect='equal')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('bte_2x1stOrd_matrix_heatmaps1.png')
plt.show()


#block6
#my best effort so far is this block which demonstrates propagation by forward euler
fig, axs = plt.subplots(figsize=(6,4))
nA = 1000 #The quantity of propagators to compute
lAc = [] #The running list of each propagator, quantity nA
lTimes = []
cmp = 3 #half of npts=6, 50% compression
#for i in range(0,len(G)-11,1):
strt = 1
major_stride = 12 #each propagator is 12 timesteps from the prior
minor_stride = 1 #not skipping timesteps within prior/posterior observations
npts = 6 #keeping the number of observations small per propagator, trying to make the propagator see the system as nearly a linear change
for i in range(nA):
    #Gt.append(G[i:11+i])
    #Again using a second order scheme for integrating
    N0 = np.array(GN[strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
    F0 = np.array(GF[strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
    #print(N0.shape,F0.shape)
    X  = np.concatenate((N0,F0),axis=0) #The priors X, are a composite of two coupled observations [N0;F0], both contributing to the prediction of Y, hence the A matrix has twice the size containing how Y evolves from each group of prior observations
    #Here Y is defined as the rate of change of observations. Alternatively could use just an observation of the system, but the physics being studied are nonlinear second order ODE (or nonlinear system of two first order ODE)
    N1 = np.array(GN[strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
    F1 = np.array(GF[strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T
    dN = (N1-N0)/minor_stride
    dF = (F1-F0)/minor_stride
    Y  = np.concatenate((dN,dF),axis=0) #The posteriors Y, are a composite of two coupled difference observations [dN/dt;dF/dt], e.g. the state of Y, hence the A matrix has 4x4 block form containing how Y evolves from each group of prior observations
    #It can be seen that the Y observation is X1-X0 while the X observation is X0. Thus the system behavior being trained is X1-X0 = A(X0). Given X0, the A propagator will yield X1-X0 and then X0. This Y observation is divided by the minor stride timestep to yield a proper first derivative approximate
    u,s,vt = svd(X,full_matrices=False) #full matrices=false, want to be able to compress, S has nonzero only
    #A = Y@v.T@np.diag(sinv)@u.T
    X_pinv = pseudo_inverse_from_svd(u, s, vt, r=cmp) # v.T[:,0:cmp]@np.diag(sinv[0:cmp])@u.T[0:cmp,:]
    Ac = Y@X_pinv #taking the 50% compression, this also seems to stabilise the approximation, maybe consider a study on the variation of this with integration accuracy
    lAc.append(Ac) #this will contain a time evolution of differential propagators
    lTimes.append(ts[strt+i*major_stride+2*minor_stride]) #here I'm sampling times at the same rate (major_stride) such that I have time positions aligned with Ac transforms
#print(lTimes[0])

#With propagators formed, here begins the forward euler loop. this is intended to operate between large strides in the pickle solutions, starting with the beginning conditions of one stride and matching the final solutions (from BTE) at the end of the stride
Nstart=npop[0].T[Linert:Mactive] #population profile from the start
Fstart=flux[0].T[Linert:Mactive] #flux profile from the start
n0=decompose_field(xcenter,Nstart) #Decompose the inside layer only, scale the temperature to natural units
f0=decompose_field(xcenter,Fstart) #Decompose the inside layer only, scale the temperature to natural units
print(n0.shape,f0.shape)
c0=np.concatenate((n0,f0),axis=0)
print(c0.shape)
nflux=np.concatenate((npop,flux),axis=0)
Nn = n0.shape[0]
#print(c0)
#Tappx = interp_temp(xcenter,c0)
#print(Tappx)
#plt.plot(xcenter ,Tappx)
#print(c)
nepoch=10 #I'm defining each epoch to be the period I'm running parallel integration alongside BTE solutions. Comparison happens at the epoch end. 
time_slice =np.linspace(100,5000,nepoch,dtype=int) #these i values are the epoch end points, so the first epoch runs 
#from i=0 to i=100, the next i=100 to i=590
#Notably I reused the list time_slice here. Previously it was returned by the loadGinterpolants(1) call and had a facile step of 1, here its being reused to define the epoch stepping

#I expect there to be some variation of forward euler with actual BTE. I'm defining here, with chatGPT guidance, a measurement operator H that is responsible for computing the measurement (Tavg) that can be compared to the BTE epoch measurement
H = np.array([1,0.,1,0.,1,0.,1,0.,1,0.,1,0.,1],dtype=float).reshape((13,1)) #Average of Legendre polynomial is the sum of even Pn coefficients, c0+c2+c4...

#This is maybe a less useful attempt. If I know what Tavg is at the end of each epoch by BTE, and the integration Tavg (by opeator H) is off by some difference, that difference (an integration quantity) needs to be distributed back to the Legendre coefficients before initiating the next epoch. This will ensure that Tavg matches between BTE and forward euler (FE) between epochs. This doesn't ensure the T profile will match. In some way should the H operator pseudoinversion matter? Or the current magnitudes of the Legendre coefficients be the weights for redistribution? The incorporation or implementation of the integration corrector is yet undefined and deserves consideration.
n = 2
spread = H/np.array([1,1,2,1,3,1,4,1,5,1,6,1,7],dtype=float).reshape((13,1))**n
Hp = spread/sum(spread)
#Hp = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=float).reshape((13,1))
#print(Hp)

#Begin integration
i=0
Navg_0 = 0.1 #Tavg at start of epoch, 0.1 is just this dataset and needs updated
t_0 = 0.

colormap = plt.get_cmap('tab10', nepoch)
colors = [colormap(k) for k in range(nepoch)] #when plotting exact vs approx curves, each epoch has own color from cmap
for t, Cs, col in zip(ts[time_slice],nflux[time_slice],colors):#[0:1]: 10 elements to loop, epochs
    Navg = np.dot(H.T,decompose_field(xcenter,Cs[:Nn].T[Linert:Mactive])) #Ts is the BTE solution and this operation with H measures the average T at the epoch start
    while lTimes[i]<t: #lTimes was recorded during formation of the propagators, this is the within-epoch integration loop that runs until the time value of the propagator reaches t, the epoch end
        '''if i<7: #debugging
            print(lAc[i])
        else:
            break'''
        #The process is as such, the prior solution n0 and f0 are stacked and multiplied with current Ac (i'th) to provide dc(:=c1-c0)
        dc = lAc[i]@c0
        #however this difference is normalized by the minor stride interval (first deriv appx), but we are counting by major strides, so the difference, dc, needs to be scaled by major_stride to predict the increase of c2 over c1 when major_stride had elapsed
        c1=c0+ major_stride*dc #e.g. +12*dc
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
        c0=c1 #update c_i solutions
        if False and i%100==0:
            #print(c2.T)
            print(np.dot(H.T,c1[Nn,:]),y)
        #print(Ac)
        #print(c)
        i+=1
    #print(i)
    plt.plot(xcenter ,interp_temp(xcenter,c1[:Nn]),color=col)
    plt.plot(xcenter ,Cs[:Nn].T[Linert:Mactive],'o',color=col,markevery=100)
    Navg_0 = Navg #Also part of an in-process corrector mechanism.
    t_0 = t
    #break
plt.xlim(0.,L)
plt.ylim(0.0,1.01)
plt.xlabel(f"x [{Lunits}]")
plt.ylabel(f"[{Nunits}]")
plt.savefig('bte_predictor_performance.png')
plt.show()

#block7
if False: #Just rying to see patterns in the transform
    fig, axs = plt.subplots(1,5,figsize=(12,4))
    #fig, ax = plt.subplots(figsize=(12,4))
    for j,ax in enumerate(axs):#[0,:]):
        im = ax.imshow(Gt[j], cmap='tab20', vmin=-.2,vmax=.2,
                origin='upper', interpolation='nearest', aspect='equal')
    '''for j,ax in enumerate(axs[1,:]):
        im = ax.imshow(Gt[j+5], cmap='tab20', vmin=-5.,vmax=2.,
                origin='upper', interpolation='nearest', aspect='equal')'''
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #plt.title(f'Green\'s Function after {Nt} steps')
    #plt.xlabel(f"x [{Lunits}]")
    #plt.ylabel(f'{Nunits}')
    plt.tight_layout()
    #plt.savefig('matrix_evolution.png')
    plt.show()


#The previous process of integrating in a second order fashion was successful because the physical equations are a 2 equation first order coupled system. The second order form just matched the reduction from two equations into one by differentiation. Here I formulate a variation wherein I maintain the coupled system. I define two different observations, N and F. 
if False:
    N0 = np.array(GN[strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
    F0 = np.array(GF[strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
    #print(X0.shape,X1.shape)
    X  = np.concatenate((N0,F0),axis=0)
    #Again using the change as the posterior, this should produce dY/dt=A(X0) or the relation between change and priors
    dN1  = (np.array(GN[strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T-N0)/minor_stride
    dF1  = (np.array(GF[strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T-F0)/minor_stride
    Y  = np.concatenate((dN1,dF1),axis=0)
    #It can be seen that the Y observation is (X1-X0)/dt while the X observations are 0'th values of fields N and F. Thus the system behavior being trained is (X1-X0)/dt = A(X0). Given X0, the A propagator will yield X1-X0 and then X1. Again this Y observation is divided by the minor stride timestep to yield a proper first derivative approximate. The value of this formulation as a tracking of two coupled fields is because the internal F field, decomposed into the GF observations, are dependent on the N field as well as the boundary values of F. I would like to extend this version one step further to contain boundary values like such:
    N0 = np.array(GN[strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
    F0 = np.array(GF[strt+i*major_stride  :strt+i*major_stride+npts*minor_stride  :minor_stride]).T
    BF = np.array([1,0,0,0])#?
    #print(X0.shape,X1.shape)
    X  = np.concatenate((N0,F0,BF),axis=0)
    #Again using the change as the posterior, this should produce dY/dt=A(X0;BF) or the relation between change and priors
    dN1  = (np.array(GN[strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T-N0)/minor_stride
    dF1  = (np.array(GF[strt+i*major_stride+minor_stride:strt+i*major_stride+npts*minor_stride+minor_stride:minor_stride]).T-F0)/minor_stride
    Y  = np.concatenate((dN1,dF1),axis=0)

    #I know there does exist a linear transform (Greens function style) such that F=G*BF. And I have demonstrated this with basic thermal fields (Fourier eq) in 2D. However in this case the domain is 1D and I am predicting dF(t)=G*BF(t). These matrices N0 and F0 are multiple prior observations that evolve into multiple posterior observations. What is BF in that space?  #The time varying greens function method applies a kernel to the input history by convolution which is also a linear process. Therefore the same method to develop the transform matrix should admit a convolution operation.
#If the input signal is FB(t) and the output of the convolution is F(t+dt), then the FB should be :=FB(dt) (?yes?) and is the forcing function being applied during the interval dt. I desire to apply it as a vector of Legendre coefficients, probably up to n=3 (or 2?). If each time step is short, the polynomials become splines across the many time solutions. Using orthogonal coordinates should permit the superposition of multiple vectors to form arbitrary input signals.
#This individual task can be developed separate of the other nonlinear process to show viability.