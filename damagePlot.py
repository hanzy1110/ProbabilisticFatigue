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
log_N = SN_data['X'].flatten().reshape((-1,1))
S = SN_data['Y'].flatten()

plt.scatter(S,log_N, label='Experimental')
plt.legend()
plt.savefig('ExperimentalData.jpg')
plt.close()
#%%
# A one dimensional column vector of inputs.
with pm.Model() as SNCurveModel:
    # Specify the covariance function.
    l = pm.HalfCauchy('lenght_scale', beta=3)
    eta = pm.HalfNormal('eta', sigma=3)
    cov_func = eta**2*pm.gp.cov.ExpQuad(input_dim=1, ls=l)

    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.Marginal(cov_func=cov_func)

    # The scale of the white noise term can be provided,
    sigma = pm.HalfCauchy("sigma", beta=5)
    y_ = gp.marginal_likelihood("y", y=S, X=log_N, noise=sigma)

    # trace = pm.sample(draws=5000, tune=2000)
#%%
ds = 10 # in some units that make sense
logNNew = np.linspace(log_N.min(), log_N.max(), 100)[:, None]
#%%
with open('plots3/samples.json','r') as file:
    y_samples = json.loads(file.read())

y_samples = {key:jnp.array(val) for key, val in y_samples.items()}

#%%
# counts, bins = jnp.histogram(y_samples['damageVals'].flatten())

# _, ax = plt.subplots(1,1,figsize=(12,8))
# ax.hist(bins[:-1], bins, weights=counts)
# ax.set_xlabel('Stress')
# ax.set_ylabel('Damage Parameter A')
# plt.savefig('plots4/damage.jpg')
# plt.close()

_, ax = plt.subplots(1,1,figsize=(12,8))
plot_gp_dist(ax, samples=y_samples['SNew'], x=logNNew)
ax.set_ylabel('S')
ax.set_xlabel('logN')
ax.scatter(log_N,S, label='Experimental Data')
ax.legend()
plt.savefig('plots4/NewDist.jpg')
plt.close()

# _, ax = plt.subplots(1,1,figsize=(12,8))
# plot_gp_dist(ax, samples=y_samples['logNNewDS'], x=SNewDS)
# ax.set_ylabel('LogN')
# ax.set_xlabel('S')
# ax.scatter(S,log_N, label='Experimental Data')
# ax.legend()
# plt.savefig('plots4/NNewDist2.jpg')
# plt.close()


# # The mean and full covariance
# mu, cov = gp.predict(SNew, point=trace[-1])

# # With noise included
# mu, var = gp.predict(SNew, point=trace[-1],  diag=True, pred_noise=True)

