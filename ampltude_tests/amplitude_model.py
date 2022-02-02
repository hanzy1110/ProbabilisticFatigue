#%%
import itertools
import pymc3 as pm
import arviz as az
import pandas as pd 
import numpy as np

from scipy import stats

from typing import Dict
import theano.tensor as tt
import matplotlib.pyplot as plt
    
def mixture_density(w, alpha, beta, scalling, x):
    logp = tt.log(w) + pm.Weibull.dist(alpha, beta).logp(x)
    return scalling * tt.sum(tt.exp(logp), axis=1)
    

data = pd.read_csv('cleansed_csvs/5BBOL2-137_VANO 136_OPGW_807625_19_02_2020.csv')
data.set_index(data['Frequency [Hz]'])

#%%
tot_cycles = np.array(data.iloc[-1,1:], dtype=np.float64)
amplitudes = np.array(list(data.columns)[1:], dtype=np.float64)

#%%
idxs = np.argsort(amplitudes)
tot_cycles = np.array([tot_cycles[idx] for idx in idxs])

amplitudes = np.hstack((np.zeros(1), amplitudes))
amplitudes = np.sort(amplitudes)

amplitudes /= amplitudes.max()
tot_cycles /= np.sum(tot_cycles)

plt.plot(amplitudes[1:], tot_cycles)

#%%
amplitudes = amplitudes[1:].reshape(-1,1)

alpha_0 = np.array([1.5, 2.6])
beta_0 = np.array([1., 1.4])
w0 =np.array([1, 1e-1])
scalling =np.array([1])

yhat = mixture_density(w0, alpha_0, beta_0, scalling, amplitudes).eval()
y = yhat + np.random.normal(loc=0, scale=.1, size=len(yhat))

plt.plot(amplitudes, yhat, label='approx')
plt.scatter(amplitudes, y, label = 'noisy')
plt.scatter(amplitudes,tot_cycles, label = 'data')
plt.legend()

#%%

with pm.Model() as disp_model:
    w = pm.HalfCauchy('w', beta=5., shape=1)
    alpha = pm.HalfNormal('alpha', sigma= 1., shape=1)
    beta = pm.HalfNormal('beta', sigma= 1., shape=1)
    scalling = pm.HalfNormal('scalling', sigma= 2., shape=1)
        
    # alpha = pm.Beta('alpha', alpha=2, beta=2, shape=1)
    # beta = pm.Beta('beta', alpha=2, beta=2, shape=1)
    noise = pm.HalfNormal('noise', sigma=1)
    
    # weibull_distro = pm.DensityDist('webull_distro', logp=log_mixture_density, observed=dict(w=w, alpha=alpha, beta=beta))
    normed_disp = pm.Normal('obs', mixture_density(w, alpha, beta, scalling, amplitudes[1:,None]), noise, observed=tot_cycles)
    
    # trace:Dict[str,np.ndarray] = pm.sample_smc()
    trace:Dict[str,np.ndarray] = pm.sample(draws=4000, chains = 4, tune=2000, target_accept=0.92)
    
    print(pm.summary(trace))
    pm.plot_trace(trace)

# # %%
# with disp_model:
#     mixture = pm.Deterministic('mixture',mixture_density(w, alpha, beta, amplitudes[1:,None]))
#     posterior_samples = pm.sample_posterior_predictive(trace, samples=24000, var_names=['mixture','obs'])
# %%
# az.plot_ppc(az.from_pymc3(posterior_predictive=posterior_samples, model = disp_model))
#%%
# max_params = 10
# params = itertools.product(*[trace["alpha"].flatten()[:max_params], trace["beta"].flatten()[:max_params], trace['w'].flatten()[:max_params]])
# _, ax = plt.subplots()
# for a, b, w in params:
#     y = mixture_density(w, a, b, amplitudes).eval()
#     ax.plot(amplitudes.reshape(1,24)[0], y, c="k", alpha=0.4)

# ax.scatter(amplitudes.reshape(1,24)[0], tot_cycles)
# ax.set_xlabel("Amplitude")
# ax.set_ylabel("Frequency Observed")
# ax.set_title("Posterior Predictive check -- Weakly regularizing priors");
# %%
alpha_m = trace["alpha"].flatten().mean()
beta_m = trace["beta"].flatten().mean()
scale = trace['scalling'].flatten().mean()
omega_m = trace['w'].flatten().mean()

_, ax = plt.subplots()

y = mixture_density(omega_m, alpha_m, beta_m, scale, amplitudes).eval()
ax.plot(amplitudes.reshape(1,24)[0], y, c="k", alpha=0.4, label = 'prediction')

ax.scatter(amplitudes.reshape(1,24)[0], tot_cycles, label = 'Data')
ax.set_xlabel("Amplitude")
ax.set_ylabel("Frequency Observed")
ax.set_title("Posterior Predictive check")
plt.legend()

# %%
