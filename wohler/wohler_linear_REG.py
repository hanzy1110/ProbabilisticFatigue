#%%
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
from pymc3.gp.util import plot_gp_dist

def basquin_rel(N, B,b):
    return B*(N**b)

B = 10
b = -1e-1

Ns = np.linspace(1e4, 1e7, 200)
sigmas = np.array([basquin_rel(val, B,b) for val in Ns])

logN = np.log(Ns)
logSigma = np.log(sigmas)
variation_coeff = 0.2

sigmas_rand = sigmas*(1+variation_coeff*np.random.normal(0, scale = .1, size = len(sigmas))) 

logSrand = np.log(sigmas_rand)

fig = plt.figure()
ax = fig.gca()
ax.plot(Ns, sigmas, label = 'mean')
ax.scatter(Ns, sigmas_rand, label = 'obs', color = 'r', marker='x')
plt.legend()

fig = plt.figure()
ax = fig.gca()
ax.plot(logN, logSigma, label = 'mean')
ax.scatter(logN, logSrand, label = 'obs', color = 'r')
plt.legend()
#%%
with pm.Model() as GUEDEs_Model:
    a = pm.Normal('a', mu=0, sigma = 10)
    A = pm.HalfNormal('A', sigma = 10)

    variation_coeff = pm.HalfCauchy('variation_coeff',beta=5)
    mean = a*logN + A
    noise = variation_coeff*mean
    
    likelihood = pm.Normal('y', mu = mean, sigma = noise, observed = logSrand)
    trace = pm.sample(4000, tune = 2000)
    
    posterior_samples = pm.sample_posterior_predictive(trace=trace, samples = 1000)
    pm.traceplot(trace)
    print(pm.summary(trace))


#%%
logSrand = np.log(sigmas_rand)

fig = plt.figure(figsize=(7, 7))
ax = fig.gca()
plot_gp_dist(ax, posterior_samples['y'], logN)
ax.plot(logN, logSrand, "x", label="data")
ax.plot(logN, logSigma, label="true regression line", lw=3.0, c="y")
ax.set_title("Posterior predictive regression lines")
ax.legend(loc=0)
ax.set_xlabel("logN")
ax.set_ylabel(r'$log \sigma$')
# %%
Ns = np.linspace(1e4, 1e7, 200)
sigmas = np.array([basquin_rel(val, B,b) for val in Ns])

logN = np.log(Ns)
logSigma = np.log(sigmas)
variation_coeff = 0.2

sigmas_rand += np.random.normal(0, scale = .1, size = len(sigmas)) 

logSrand = np.log(sigmas_rand)

fig = plt.figure()
ax = fig.gca()
ax.plot(Ns, sigmas, label = 'mean')
ax.scatter(Ns, sigmas_rand, label = 'obs', color = 'r', marker='x')
plt.legend()

fig = plt.figure()
ax = fig.gca()
ax.plot(logN, logSigma, label = 'mean')
ax.scatter(logN, logSrand, label = 'obs', color = 'r')
plt.legend()
#%%
with pm.Model() as ESOPE_Model:
    a = pm.Normal('a', mu=0, sigma = 10)
    A = pm.HalfNormal('A', sigma = 10)

    mean = a*logN + A
    noise = pm.HalfNormal('noise', sigma = 10)
    
    likelihood = pm.Normal('y', mu = mean, sigma = noise, observed = logSrand)
    trace = pm.sample(4000, tune = 2000)
    
    posterior_samples = pm.sample_posterior_predictive(trace=trace, samples = 1000)
    pm.traceplot(trace)
    print(pm.summary(trace))


#%%
logSrand = np.log(sigmas_rand)

fig = plt.figure(figsize=(7, 7))
ax = fig.gca()
plot_gp_dist(ax, posterior_samples['y'], logN)
ax.plot(logN, logSrand, "x", label="data")
ax.plot(logN, logSigma, label="true regression line", lw=3.0, c="y")
ax.set_title("Posterior predictive regression lines")
ax.legend(loc=0)
ax.set_xlabel("logN")
ax.set_ylabel(r'$log \sigma$')
#%%