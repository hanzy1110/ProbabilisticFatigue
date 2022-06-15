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

ds = 2 # in some units that make sense
logNNew = np.linspace(log_N.min(), log_N.max(), 100)[:, None]

#%%

with pm.Model() as SNCurveModel:
    l = pm.HalfCauchy('lenght_scale', beta=1, shape=(2,))
    eta = pm.HalfCauchy('eta', beta=1, shape=(2,))
    cov_func = eta[1]**2 * pm.gp.cov.Exponential(1, ls=l[1])
    gp = pm.gp.Latent(cov_func=cov_func)
    muF = gp.prior('muF', X=log_N)

    # l_sigma = pm.Gamma('l_sigma', alpha=7, beta=0.5)
    # eta_sigma = pm.Gamma('eta_sigma', alpha=7, beta=0.5)
    # cov_sigma = eta[0]**2 * pm.gp.cov.Exponential(1, ls=l[0])
    # gp_sigma = pm.gp.Latent(cov_func=cov_sigma)
    # log_sigmaF = gp_sigma.prior('log_sigmaF', log_N)
    log_sigmaF = pm.Weibull('log_sigmaF', alpha=1, beta=1)
    nu = pm.Gamma('nu', alpha=8, beta=0.5)
    likelihood = pm.StudentT('likelihood',
                             nu=nu, mu=muF,
                             sigma=log_sigmaF,
                             observed=S)

    # trace = pm.sample(draws=5000, tune=2000)
    trace = pm.sample(return_inferencedata=False)
    summ = az.summary(trace)
    print(summ)
    az.plot_trace(trace)
    plt.savefig('plots3/traceplot.jpg')
    plt.close()

#%%
with SNCurveModel:
    SNew = gp.conditional("SNew", Xnew=logNNew)
    # sigma_ = gp_sigma.conditional('sigma', Xnew=logNNew)
    # or to predict the GP plus noise
    y_samples = pm.sample_posterior_predictive(trace=trace, var_names=['SNew'])

#%%

# def A(logNNew):
#     pass
# damageFun = vmap(A,in_axes=(0,0))
# damageVals = damageFun(y_samples['NNew'], y_samples['NNewDS'])
# y_samples['damageVals'] = damageVals

# y_samples = {key: val.tolist() for key, val in y_samples.items()}

# y_samples['damageVals'] = damageVals

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

