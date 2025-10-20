import numpy as np
from scipy.interpolate import CubicSpline


class Propagator:
    def __init__(self):
        """
        Parameters
        ----------
        t_list: sorted times at which A_list is defined
        A_list: list or array of A matrices, A_list[k] corresponds to t_list[k]
        A_of_t : callable
            Function A_of_t(t) -> matrix A(t) at the given time
        rhs : callable
            the right hand side of the differential equation to be integrated
        I'm leaving the init function general for overloading purposes, will probably update later for a error-checking keyword instantiation
        """
        self.t_list = None
        self.A_list = None
        self.major_stride = None
        self.splines = None
        self.dim0 = None;self.dim1=None
        

    def buildMe(self,times,A_mats):
        # Build cubic spline for each element
        self.dim0,self.dim1 = A_mats.shape[1:3]
        self.splines = [[CubicSpline(times, A_mats[:,i,j]) for j in range(self.dim1)] for i in range(self.dim0)]

    def A_of_t(self,t):
        return np.array([[self.splines[i][j](t) for j in range(self.dim1)] for i in range(self.dim0)])
    
    def rhs(self, t, c):
        return (self.A_of_t(t)@c*self.major_stride/(self.t_list[1]-self.t_list[0])).reshape(-1)  #(26,)
