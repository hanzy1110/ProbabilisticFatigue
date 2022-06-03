#%%
import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

from typing import Dict
import theano.tensor as tt
import matplotlib.pyplot as plt

def hist_sample(hist, n):
    """Genertae histogram sample
    Args:
        hist (array): hist[0]: frecuencies/probs of X values, hist[1]: X values
        n ([type]): number of samples
    Returns:
        [list]: list with samples
    """
    return np.random.choice(hist[1], size=n, p=hist[0]/sum(hist[0]))

data = pd.read_csv('800369.csv')
cycles = np.array(data.iloc[-1].values[1:], dtype=np.float64)
b = -0.1
#%%
amplitudes = np.array(list(data.columns)[1:], np.float64)
amplitudes_sample = hist_sample([cycles, amplitudes], n=2500)
# %%

# frequency = np.array(data['Frequency [Hz]'].values, dtype=np.float64).reshape(-1,1 )
_,ax = plt.subplots(1,1, sharex=False, figsize=(10,8))
ax.hist(amplitudes_sample, density=True)
ax.set_xlabel('Amplitudes')
ax.set_ylabel('Density')
plt.savefig('plots2/amplitudes.jpg')
plt.close()
# %%
amplitudes_sample/=amplitudes_sample.max()
with pm.Model() as amplitude_model:
    alpha = pm.HalfNormal('alpha', sigma=10)
    beta = pm.HalfNormal('beta', sigma=10)
    # lam = pm.HalfCauchy('lam', beta=4)
    # likelihood = pm.Exponential('likelihood',lam=lam,
    #                         observed=amplitudes_sample)
    likelihood = pm.Weibull('likelihood',alpha=alpha, beta=beta,
                            observed=amplitudes_sample)
    trace = pm.sample(draws=3000,
                      chains=3,
                      tune=3000,
                      # target_accept=0.97,
                      return_inferencedata=False
                      )
    print(az.summary(trace))
    az.plot_trace(trace)
    plt.savefig('plots2/traceplot.jpg')
    plt.close()
    az.plot_posterior(trace)

    plt.savefig('plots2/posteriorArviz.jpg')
    plt.close()
# %%
with amplitude_model:
    samples = pm.sample_posterior_predictive(trace,6000, var_names=['likelihood'])

_,ax = plt.subplots(1,1, sharex=False, figsize=(12,5))
# sns.kdeplot(amplitude_model, ax=ax, label='observed')
ax.hist(amplitudes_sample, density=True)
sns.kdeplot(samples['likelihood'].flatten(), ax=ax, label='Inference')
plt.legend()
plt.savefig('plots2/posterior.jpg')
plt.close()
# %%
