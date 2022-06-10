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

def logistic(x, a, x0, c):
    # a is the slope, x0 is the location
    return c *(1 - pm.math.invlogit(a * (x - x0)))

#%%
# A one dimensional column vector of inputs.
with pm.Model() as SNCurveModel:
    # Specify the covariance function.
    l = pm.HalfCauchy('lenght_scale', beta=3)
    eta = pm.HalfNormal('eta', sigma=3)
    x0 = pm.HalfNormal('x0', sigma=10)
    # a = pm.Normal('a', mu=0, sigma=10)
    # c = pm.Normal('c', mu=0, sigma=10)

    cov_func_base = eta**2*pm.gp.cov.Mattern52(input_dim=1, ls=l)
    cov_func = pm.gp.cov.ScaledCov(1, scaling_func=logistic, args=(a,x0,c), cov_func=cov_func_base)
    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.Marginal(cov_func=cov_func)

    # The scale of the white noise term can be provided,
    sigma = pm.HalfCauchy("sigma", beta=5)
    y_ = gp.marginal_likelihood("y", X=S, y=log_N, noise=sigma)

    # trace = pm.sample(draws=5000, tune=2000)
    trace = pm.sample(return_inferencedata=False)
    summ = az.summary(trace)
    print(summ)
    az.plot_trace(trace)
    plt.savefig('plots3/traceplot.jpg')
    plt.close()

#%%
ds = 10 # in some units that make sense
SNew = np.linspace(S.min(), S.max(), 100)[:, None]
SNewDS = np.linspace(S.min()+ds, S.max()+ds, 100)[:, None]
with SNCurveModel:
    NNew = gp.conditional("NNew", Xnew=SNew, pred_noise=True)
    NNewDS = gp.conditional("NNewDS", Xnew=SNewDS, pred_noise=True)
#%%
with SNCurveModel:

    # or to predict the GP plus noise
    y_samples = pm.sample_posterior_predictive(trace=trace, var_names=['NNew','NNewDS'])

#%%
def A(arr1, arr2):
    arr1 = jnp.array(arr1)
    arr2 = jnp.array(arr2)
    return arr1/arr2

damageFun = vmap(A,in_axes=(0,0))
damageVals = damageFun(y_samples['NNew'], y_samples['NNewDS'])
y_samples['damageVals'] = damageVals

y_samples = {key: val.tolist() for key, val in y_samples.items()}

with open('plots3/samples.json','w') as file:
    json.dump(y_samples, file)
#%%
# _, ax = plt.subplots(1,1,figsize=(12,8))
# ax.hist(damageVals.flatten(), density=True)

# ax.set_xlabel('Stress')
# ax.set_ylabel('Damage Parameter A')
# plt.savefig('plots3/damage.jpg')
# plt.close()

# _, ax = plt.subplots(1,1,figsize=(12,8))
# plot_gp_dist(ax, samples=y_samples['NNew'], x=NNew)
# plt.savefig('plots3/NNewDist.jpg')
# plt.close()

# _, ax = plt.subplots(1,1,figsize=(12,8))
# plot_gp_dist(ax, samples=y_samples['NNewDS'], x=NNewDS)

# plt.savefig('plots3/NNewDist2.jpg')
# plt.close()


# # # The mean and full covariance
# # mu, cov = gp.predict(SNew, point=trace[-1])

# # # With noise included
# # mu, var = gp.predict(SNew, point=trace[-1],  diag=True, pred_noise=True)

