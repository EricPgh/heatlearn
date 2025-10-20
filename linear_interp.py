import numpy as np

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
    def buildMe(self,times,A_mats):
        self.t_list = times
        self.A_list = A_mats
        
    def A_of_t(self,t):
        idx = np.searchsorted(self.t_list, t) - 1
        if idx < 0: return self.A_list[0]
        if idx >= len(self.t_list)-1: return self.A_list[-1]
        t0 = self.t_list[idx]; t1 = self.t_list[idx+1]
        a = (t - t0) / (t1 - t0) if t1>t0 else 0.0
        return (1-a)*self.A_list[idx] + a*self.A_list[idx+1]
    
    def rhs(self, t, c):
        return (self.A_of_t(t)@c*self.major_stride/(self.t_list[1]-self.t_list[0])).reshape(-1)  #(26,)

