#%%
import numpy as np
import scipy
import pymc3 as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from pymc3.gp.util import plot_gp_dist
import json
import jax.numpy as jnp
from jax import vmap ,jit

def calculateA(s, Ssamples, N):
    pass


SN_data = scipy.io.loadmat('data/SN_curve.mat')
log_N = SN_data['X'].flatten().reshape((-1,1))
S = SN_data['Y'].flatten()
ds = 2 # in some units that make sense
logNNew = np.linspace(log_N.min(), log_N.max(), 100)[:, None]
with open('plots3/samples.json','w') as file:
    y_samples = json.load(file)

y_samples = {key:jnp.array(val) for key, val in y_samples.items()}

