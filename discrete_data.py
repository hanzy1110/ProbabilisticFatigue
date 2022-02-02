#%%
import pymc3 as pm 
import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 100)
f = lambda x: np.exp(x)*4.+2.5
plt.scatter(x, f(x)+np.random.randn(100))

#%%
obs = f(x)+np.random.randn(100)
with pm.Model():
    slope = pm.Normal('b', 0., 10.)
    intercept = pm.Normal('a', 0., 10.)
    pm.Normal('obs', tt.exp(x)*slope + intercept, 1, observed=obs)
    trace = pm.sample()
    
pm.traceplot(trace)

#%%
with pm.Model():
    slope = pm.Normal('b', 0., 10.)
    intercept = pm.Normal('a', 0., 10.)
    sigma = pm.HalfCauchy('sd', 5.)
    pm.Normal('obs', tt.exp(x)*slope + intercept, sigma, observed=obs)
    trace = pm.sample()
    
pm.traceplot(trace)

#%%
from pymc3.math import logsumexp

def mixture_density(w, mu, sd, x):
    logp = tt.log(w) + pm.Normal.dist(mu, sd).logp(x)
    return tt.sum(tt.exp(logp), axis=1)

x = np.linspace(-5, 10, 100)[:, None]
mu0 = np.array([-1., 2.6])
sd0 = np.array([.5, 1.4])
w0 =np.array([10, 60])
yhat = mixture_density(w0, mu0, sd0, x).eval()
y = yhat + np.random.randn(100)

plt.plot(x, yhat)
plt.scatter(x, y)

#%%

with pm.Model():
    w = pm.HalfNormal('w', 10., shape=2)
    mu = pm.Normal('mu', 0., 100., shape=2)
    sd = pm.HalfCauchy('sd', 5., shape=2)
#     noise = pm.HalfCauchy('eps', 5.)
    pm.Normal('obs', mixture_density(w, mu, sd, x), 1., observed=y)
    trace = pm.sample()
    
pm.traceplot(trace)

#%%
with pm.Model():
    w = pm.HalfNormal('w', 10., shape=2)
    mu = pm.Normal('mu', 0., 100., shape=2)
    sd = pm.HalfCauchy('sd', 5., shape=2)
    noise = pm.HalfNormal('eps', 5.)
    pm.Normal('obs', mixture_density(w, mu, sd, x), noise, observed=y)
    trace = pm.sample()
    
pm.traceplot(trace)