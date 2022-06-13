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
NMax = log_N.max()
SMax = S.max()
log_N /= NMax
S/=SMax
ds = 10 # in some units that make sense
SNew = np.linspace(S.min(), S.max(), 100)[:, None]

ℓ_μ, ℓ_σ = [stat for stat in get_ℓ_prior(S.flatten())]

def logistic(x, a, x0, c):
    # a is the slope, x0 is the location
    return c *(1 - pm.math.invlogit(a * (x - x0)))

#%%
# A one dimensional column vector of inputs.
with pm.Model() as SNCurveModel:
    ℓ = pm.InverseGamma("ℓ", mu=ℓ_μ, sigma=ℓ_σ)
    η = pm.Gamma("η", alpha=2, beta=1)
    cov = η ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ℓ) + pm.gp.cov.WhiteNoise(sigma=1e-6)

    gp_ht = pm.gp.Latent(cov_func=cov)
    μ_f = gp_ht.prior("μ_f", X=S)

    σ_ℓ = pm.InverseGamma("σ_ℓ", mu=ℓ_μ, sigma=ℓ_σ)
    σ_η = pm.Gamma("σ_η", alpha=2, beta=1)
    σ_cov = σ_η ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=σ_ℓ) + pm.gp.cov.WhiteNoise(sigma=1e-6)

    σ_gp = pm.gp.Latent(cov_func=σ_cov)
    lg_σ_f = σ_gp.prior("lg_σ_f", X=S)
    σ_f = pm.Deterministic("σ_f", pm.math.exp(lg_σ_f))

    lik_ht = pm.Normal("lik_ht", mu=μ_f, sd=σ_f, observed=log_N)

    # trace = pm.sample(target_accept=0.95, chains=2, return_inferencedata=True, random_seed=2022)
    # summ = az.summary(trace)
    # print(summ)
    # az.plot_trace(trace)
    # plt.savefig('plots3/traceplot.jpg')
    # plt.close()

#%%
with open('plots3/samples.json','r') as file:
    y_samples = json.load(file)

y_samples = {key:jnp.array(val) for key, val in y_samples.items()}
# counts, bins = jnp.histogram(y_samples['damageVals'].flatten())

# _, ax = plt.subplots(1,1,figsize=(10, 4))
# ax.hist(bins[:-1], bins, weights=counts)
# plt.savefig('plots3/damageParam.jpg')
# plt.close()

_, axs = plt.subplots(1, 3, figsize=(18, 4))
# μ_samples = y_samples["NNew"].mean(axis=0)
μ_samples = y_samples["NNew"]
σ_samples = np.exp(y_samples["log_σ_f_pred"])
plot_mean(axs[0], μ_samples, Xnew=SNew, ynew=y_samples["NNew"].mean(axis=0), X=S, y=log_N)
plot_var(axs[1], σ_samples ** 2, X=S, Xnew=SNew,y_err=1)
plot_total(axs[2], μ_samples,
           var_samples=σ_samples ** 2,
           Xnew=SNew, ynew=y_samples['NNew'].mean(axis=0),
           X_obs=S, y_obs_=log_N)
plt.savefig('plots4/heteroModel.jpg')
plt.close()
