#%%
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt

def basquin_rel(N, B,b):
    return B*(N**b)

B = 10
b = -1e-1

Ns = np.linspace(1, 1e7, 200)
sigmas = np.array([basquin_rel(val, B,b) for val in Ns])

sigmas_rand = sigmas + np.random.normal(0, scale = .1, size = len(sigmas))

plt.plot(Ns, sigmas, label = 'mean')
plt.scatter(Ns, sigmas_rand, label = 'obs', color = 'r')

plt.legend()

#%%
nobs = 100
N_obs = Ns[2]
sigma_obs = basquin_rel(N_obs, B,b) + np.random.normal(0,1, size = nobs)

with pm.Model() as model_:
    B = pm.Normal('B',mu = 10, sigma = 1)
    b_ = pm.Beta('b', alpha=2, beta=5)
    b = -1 * b_
    sigma_SN = pm.HalfCauchy('sigma_SN',beta=5)

    mu_SN = B*N_obs**b

    sigma = pm.Normal('sigma_obs',mu = mu_SN, sigma = sigma_SN, observed = sigma_obs)

    trace = pm.sample(2500, tune = 2000)
    pm.traceplot(trace)
    pm.summary(trace)
# %%

