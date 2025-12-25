import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.linalg import svd,eig
import math
bPlot=False
bPlotEig=False
from FluxLearner_lowmemory import FluxLearner
FL = FluxLearner()
from FluxLearner_lowmemory import decompose_field, interp_temp, loadGinterpolants
from FluxLearner_lowmemory import pseudo_inverse_from_svd
#FL.digest_boundary_fields()
#FL.digest_fields()
with open(f'FL.pkl','wb') as file:
    pickle.dump(FL,file)
print('pickle written')
