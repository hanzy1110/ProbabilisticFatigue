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

ds = 10 # in some units that make sense
SNew = np.linspace(S.min(), S.max(), 100)[:, None]
SNewDS = np.linspace(S.min()+ds, S.max()+ds, 100)[:, None]

ℓ_μ, ℓ_σ = [stat for stat in get_ℓ_prior(S.flatten())]

def logistic(x, a, x0, c):
    # a is the slope, x0 is the location
    return c *(1 - pm.math.invlogit(a * (x - x0)))

#%%
# A one dimensional column vector of inputs.
with pm.Model() as SNCurveModel:
    ℓ = pm.InverseGamma("ℓ", mu=ℓ_μ, sigma=ℓ_σ)
    # ℓ = pm.Normal('ℓ', mu=100, sigma=10)
    η = pm.Gamma("η", alpha=20, beta=10)
    # η = pm.Normal("η", mu=100, sigma=10)

    x0 = pm.Gamma("x0", alpha=2, beta=1)
    a = pm.Gamma("a", alpha=2, beta=1)
    cov_base = η ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ℓ) + pm.gp.cov.WhiteNoise(sigma=1e-4)
    cov = pm.gp.cov.ScaledCov(1, scaling_func=logistic, args=(a, x0, 1), cov_func=cov_base)
    gp_ht = pm.gp.Latent(cov_func=cov)
    μ_f = gp_ht.prior("μ_f", X=S)

    σ_ℓ = pm.InverseGamma("σ_ℓ", mu=ℓ_μ, sigma=ℓ_σ)
    σ_η = pm.Gamma("σ_η", alpha=2, beta=1)
    σ_cov = σ_η ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=σ_ℓ) + pm.gp.cov.WhiteNoise(sigma=1e-6)

    σ_gp = pm.gp.Latent(cov_func=σ_cov)
    lg_σ_f = σ_gp.prior("lg_σ_f", X=S)
    σ_f = pm.Deterministic("σ_f", pm.math.exp(lg_σ_f))

    lik_ht = pm.Normal("lik_ht", mu=μ_f, sd=σ_f, observed=log_N)

    trace = pm.sample(target_accept=0.95, chains=2, return_inferencedata=True, random_seed=2022)
    summ = az.summary(trace)
    print(summ)
    az.plot_trace(trace)
    plt.savefig('plots3/traceplot.jpg')
    plt.close()

#%%
try:
    with SNCurveModel:
        NNew = gp_ht.conditional("NNew", Xnew=SNew)
        NNewDS = gp_ht.conditional("NNewDS", Xnew=SNewDS)
        lg_σ_f_pred = σ_gp.conditional("log_σ_f_pred", Xnew=SNew)
        # or to predict the GP plus noise
        y_samples = pm.sample_posterior_predictive(trace=trace, var_names=['NNew','NNewDS', 'log_σ_f_pred'])
except Exception as e:
    print(e)

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
