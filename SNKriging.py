#%%
import numpy as np
import scipy
import pymc3 as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from pymc3.gp.util import plot_gp_dist

SN_data = scipy.io.loadmat('SN_curve.mat')
log_N = SN_data['X'].flatten().reshape((-1,1))
S = SN_data['Y'].flatten()

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
    y_ = gp.marginal_likelihood("y", X=log_N, y=S, noise=sigma)

    trace = pm.sample(draws=5000, tune=2000)
    summ = az.summary(trace)
    print(summ)
    az.plot_trace(trace)
    plt.savefig('plots3/traceplot.jpg')
    plt.close()
# vector of new X points we want to predict the function at
newLogN = np.linspace(log_N.min(), log_N.max(), 100)[:, None]
with SNCurveModel:
    f_star = gp.conditional("f_star", Xnew=newLogN)

    # or to predict the GP plus noise
    y_star = gp.conditional("y_star", Xnew=newLogN, pred_noise=True)
    y_samples = pm.sample_posterior_predictive(trace=trace, var_names=['y_star'])
# The mean and full covariance
mu, cov = gp.predict(newLogN, point=trace[-1])

# The mean and variance (diagonal of the covariance)
mu, var = gp.predict(newLogN, point=trace[-1],  diag=True)

# With noise included
mu, var = gp.predict(newLogN, point=trace[-1],  diag=True, pred_noise=True)

fig = plt.figure(figsize=(12, 5))
ax = fig.gca()

# posterior predictive distribution
plot_gp_dist(ax, y_samples["y_star"], newLogN, plot_samples=False, palette="bone_r")

# overlay a scatter of one draw of random points from the
#   posterior predictive distribution
plt.plot(newLogN, y_samples["y_star"][800, :].T, "co", ms=2, label="Predicted data")

# plot original data and true function
plt.plot(log_N, S, "ok", ms=3, alpha=1.0, label="observed data")

plt.xlabel("logN")
plt.title("posterior predictive distribution, S*")
plt.legend()
plt.savefig('plots3/gpdist.jpg')
# draw plot
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()

sd = np.sqrt(var)
# plot mean and 2σ intervals
plt.plot(newLogN, mu, "r", lw=2, label="mean and 2σ region")
plt.plot(newLogN, mu + 2 * sd, "r", lw=1)
plt.plot(newLogN, mu - 2 * sd, "r", lw=1)
plt.fill_between(newLogN.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)

# plot original data and true function
plt.plot(log_N, S, "ok", ms=3, alpha=1.0, label="observed data")

plt.xlabel("x")
plt.title("predictive mean and 2σ interval")
plt.legend()

plt.savefig('plots3/muVar.jpg')
