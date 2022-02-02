#%%
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt

def basquin_rel(N, B,b):
    return B*(N**b)

B = 10
b = -1e-1

Ns = np.linspace(1e4, 1e7, 200)
sigmas = np.array([basquin_rel(val, B,b) for val in Ns])

logN = np.log(Ns)
logSigma = np.log(sigmas)

sigmas_rand = sigmas + np.random.normal(0, scale = .1, size = len(sigmas))

logSrand = np.log(sigmas_rand)

fig = plt.figure()
ax = fig.gca()
ax.plot(Ns, sigmas, label = 'mean')
ax.scatter(Ns, sigmas_rand, label = 'obs', color = 'r')
plt.legend()

fig = plt.figure()
ax = fig.gca()
ax.plot(logN, logSigma, label = 'mean')
ax.scatter(logN, logSrand, label = 'obs', color = 'r')
plt.legend()

#%%
with pm.Model() as model_:
    a_ = pm.HalfNormal('a', sigma = 10)
    A = pm.HalfNormal('A', sigma = 10)
    a = 1*a_

    mu = pm.gp.mean.Linear(coeffs=a, intercept=A)
    
    h = pm.HalfNormal('h', sigma = 10)
    eta = pm.HalfNormal('eta', sigma = 10)

    cov = eta**2 * pm.gp.cov.ExpQuad(1, ls = h)
    sigma = pm.HalfNormal('sigmaSN', sigma = 10)

    gp = pm.gp.Marginal(mean_func= mu , cov_func = cov)

    y = gp.marginal_likelihood('sigma_OBS', logN[:,None], logSigma, noise = sigma)

    trace = pm.sample(500, tune = 100)
    pm.traceplot(trace)
    print(pm.summary(trace))


# %%

logNnew = np.log(np.linspace(1e3,1e7, 200))[:,None]
with model_:
    f_pred = gp.conditional("f_pred", logNnew)

# To use the MAP values, you can just replace the trace with a length-1 list with `mp`
with model_:
    pred_samples = pm.sample_posterior_predictive(trace, samples=500, var_names = ['f_pred'])