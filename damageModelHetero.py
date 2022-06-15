#%%
import numpy as np
import scipy
import pymc3 as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from pymc3.gp.util import plot_gp_dist
import json

from src.plotUtils import plot_mean, plot_var, plot_total, get_ℓ_prior

import jax.numpy as jnp
from jax import vmap ,jit

SN_data = scipy.io.loadmat('data/SN_curve.mat')
log_N = SN_data['X'].flatten()
S = SN_data['Y'].flatten().reshape((-1,1))

#Transform it to N
NMax = log_N.max()
log_N /= NMax
# log_N = np.exp(log_N)
SMax = S.max()
#Normalize both values
S/=SMax
SNew = np.linspace(S.min(), S.max(), 100)[:, None]

_,ax = plt.subplots(1,1, figsize=(12,8))
ax.scatter(S,log_N)
ax.set_ylabel('N/NMax')
ax.set_xlabel('S/Smax')
plt.savefig('plots4/experimental.jpg')
plt.close()
ℓ_μ, ℓ_σ = [stat for stat in get_ℓ_prior(S.flatten())]

def logistic(x, a, x0, c):
    # a is the slope, x0 is the location
    return c *(1 - pm.math.invlogit(a * (x - x0)))

#%%
# A one dimensional column vector of inputs.
with pm.Model() as SNCurveModel:
    ℓ = pm.Gamma("ℓ", mu=ℓ_μ, sigma=ℓ_σ)
    # ℓ = pm.Normal('ℓ', mu=100, sigma=10)
    η = pm.Gamma("η", alpha=2, beta=1)
    # η = pm.Normal("η", mu=100, sigma=10)

    x0 = pm.Gamma("x0", alpha=2, beta=1)
    a = pm.Gamma("a", alpha=2, beta=1)
    c = pm.Gamma("c", alpha=2, beta=1)
    cov_base = η ** 2 * pm.gp.cov.Exponential(input_dim=1, ls=ℓ) + pm.gp.cov.WhiteNoise(sigma=1e-6)
    cov = pm.gp.cov.ScaledCov(1, scaling_func=logistic, args=(a, x0, c), cov_func=cov_base)
    gp_ht = pm.gp.Latent(cov_func=cov)
    μ_f = gp_ht.prior("μ_f", X=S)

    σ_ℓ = pm.Gamma("σ_ℓ", mu=ℓ_μ, sigma=ℓ_σ)
    σ_η = pm.Gamma("σ_η", alpha=2, beta=1)
    σ_cov = σ_η ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=σ_ℓ) + pm.gp.cov.WhiteNoise(sigma=1e-6)

    σ_gp = pm.gp.Latent(cov_func=σ_cov)
    σ_f = σ_gp.prior("lg_σ_f", X=S)
    σ_f = pm.Deterministic("σ_f", pm.math.exp(σ_f))

    nu = pm.Gamma("nu", alpha=2, beta=1)
    lik_ht = pm.StudentT("lik_ht",nu = nu,  mu=μ_f, sd=σ_f, observed=log_N)

    # trace = pm.sample(target_accept=0.95, chains=2, return_inferencedata=True, random_seed=2022)
    trace = pm.sample_smc(draws=2000, parallel=True)
    summ = az.summary(trace)
    print(summ)
    az.plot_trace(trace)
    plt.savefig('plots4/traceplot.jpg')
    plt.close()

#%%
with SNCurveModel:
    NNew = gp_ht.conditional("NNew", Xnew=SNew)
    lg_σ_f_pred = σ_gp.conditional("log_σ_f_pred", Xnew=SNew)
    # or to predict the GP plus noise
    y_samples = pm.sample_posterior_predictive(trace=trace, var_names=['NNew','log_σ_f_pred'])

#%%
# def A(arr1, arr2):
#     arr1 = jnp.array(arr1)
#     arr2 = jnp.array(arr2)
#     return arr1/arr2

# damageFun = vmap(A,in_axes=(0,0))
# damageVals = damageFun(y_samples['NNew'], y_samples['NNewDS'])
# y_samples['damageVals'] = damageVals
try:
    y_samples = {key: val.tolist() for key, val in y_samples.items()}
except Exception as e:
    print(e)

with open('plots3/samples.json','w') as file:
    json.dump(y_samples, file)
