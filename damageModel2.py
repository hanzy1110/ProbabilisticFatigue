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

SN_data = scipy.io.loadmat('data/SN_curve.mat')
log_N = SN_data['X'].flatten()
S = SN_data['Y'].flatten().reshape((-1,1))

# S = S[log_N<200]
# log_N = log_N[log_N<200]

SNEw = np.linspace(S.min(), S.max(), 100)[:, None]

def logistic(x, a, x0, c):
    # a is the slope, x0 is the location
    return c *(1 - pm.math.invlogit(a * (x - x0)))

#%%
# A one dimensional column vector of inputs.
with pm.Model() as SNCurveModel:
    l = pm.Gamma('lenght_scale', alpha=10, beta=1)
    eta = pm.HalfNormal('eta', sigma=3)

    # x0 = pm.HalfNormal('x0', sigma=10)
    # a = pm.Normal('a', mu=0, sigma=10)
    # c = pm.Weibull('c', alpha=2, beta=10)

    cov_func = eta**2 * pm.gp.cov.ExpQuad(1,ls=l)
    # cov_func = pm.gp.cov.ScaledCov(1, cov_base, logistic, (a, x0, c))
    gp = pm.gp.Marginal(cov_func=cov_func)

    sigma = pm.Gamma('sigma', alpha=10, beta=1)
    y_ = gp.marginal_likelihood("y", X=S, y=log_N, noise=sigma)

    # trace = pm.sample(draws=5000, tune=2000)
    # trace = pm.sample(target_accept=0.95, return_inferencedata=False)
    trace = pm.sample_smc(2000, chains=4, parallel=True)
    summ = az.summary(trace)
    print(summ)
    az.plot_trace(trace)
    plt.savefig('plots3/traceplot.jpg')
    plt.close()

#%%
with SNCurveModel:
    logNNew = gp.conditional("logNNew", Xnew=SNEw)
    # or to predict the GP plus noise
    y_samples = pm.sample_posterior_predictive(trace=trace, var_names=['logNNew'])

#%%
# def A(arr1, arr2):
#     arr1 = jnp.array(arr1)
#     arr2 = jnp.array(arr2)
#     return jnp.where(arr1>0, arr1, 1)/jnp.where(arr2>0, arr2, 1)

# damageFun = vmap(A,in_axes=(0,0))
# damageVals = damageFun(y_samples['logNNew'], y_samples['logNNewDS'])
# y_samples['damageVals'] = damageVals

y_samples = {key: val.tolist() for key, val in y_samples.items()}

# y_samples['damageVals'] = damageVals.tolist()

# y_samples = {key: val.tolist() for key, val in y_samples.items()}
with open('plots3/samples.json','w') as file:
    json.dump(y_samples, file)
