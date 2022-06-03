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

def equivalent_fatigue(freq, data, amplitudes):
    cycles = np.array(data.loc[data['Frequency [Hz]'] == freq], dtype=np.float64)[0][1:]
    basquin_stress = amplitudes**b
    return np.dot(cycles, basquin_stress)

def get_freq_density(freq, data):
    cycles = np.array(data.loc[data['Frequency [Hz]'] == freq], dtype=np.float64)
    return cycles.sum()

data = pd.read_csv('800356.csv')
# data['Frequency [Hz]'] = np.array(data['Frequency [Hz]'].values, dtype=np.int64)
data.set_index(data['Frequency [Hz]'], inplace=True)
data.drop(data.index.values[-1], inplace=True)

N_eq = 1e7
b = -0.1
#%%
freq_prev = np.array(data['Frequency [Hz]'].values)
amplitudes = np.array(list(data.columns)[1:], np.float64)
equivalent_cycles = np.fromiter(map(lambda x: equivalent_fatigue(x, data=data, amplitudes=amplitudes), freq_prev),
                                dtype=np.float64)
equivalent_cycles = (equivalent_cycles/N_eq) ** b
frequency_density = np.fromiter(map(lambda x: get_freq_density(x, data=data), freq_prev), dtype=np.float64)
frequency = hist_sample([frequency_density, freq_prev], n=2500)
# %%

# frequency = np.array(data['Frequency [Hz]'].values, dtype=np.float64).reshape(-1,1 )
_,ax = plt.subplots(2,1, sharex=False, figsize=(10,8))
ax[0].hist(equivalent_cycles, density=True)
ax[0].set_xlabel('Equiv. Cycles')
ax[0].set_ylabel('Density')
ax[1].hist(freq_prev)
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Density')
plt.savefig('plots2/data.jpg')
plt.close()
_,ax = plt.subplots(1,1, sharex=False, figsize=(10,8))
ax.scatter(freq_prev, equivalent_cycles)
ax.set_xlabel('Frequency')
ax.set_ylabel('Equiv Cycles')
plt.savefig('plots2/freqVsEquivC.jpg')
plt.close()
# %%

frequency = np.array(frequency, dtype=np.float64)/frequency.max()

with pm.Model() as frequency_model:
    w = pm.Dirichlet('w', np.ones(2))
    # w = np.array([1,0])
    alphas = pm.HalfNormal('alphas', sigma=10, shape=(2,))
    betas = pm.HalfNormal('betas', sigma=10, shape=(2,))

    weib_0 = pm.Weibull.dist(alpha=alphas[0], beta=betas[0])
    weib_1 = pm.Weibull.dist(alpha=alphas[1], beta=betas[1])
    GMM = pm.Mixture('likelihood',
                     w=w,comp_dists=[weib_0, weib_1],
                     observed=frequency)
    trace = pm.sample_smc(draws=2000,chains=2,parallel=False)
    # trace = pm.sample(draws=3000,
    #                   chains=3,
    #                   tune=3000,
    #                   # target_accept=0.97,
    #                   return_inferencedata=False
    #                   )
    print(az.summary(trace))
    az.plot_trace(trace)
    plt.savefig('plots2/traceplot.jpg')
    plt.close()
    az.plot_posterior(trace)

    plt.savefig('plots2/posteriorArviz.jpg')
    plt.close()
# %%
with frequency_model:
    samples = pm.sample_posterior_predictive(trace,6000, var_names=['likelihood'])

_,ax = plt.subplots(1,1, sharex=False, figsize=(12,5))
sns.kdeplot(frequency, ax=ax, label='observed')
sns.kdeplot(samples['likelihood'].flatten(), ax=ax, label='Inference')
plt.legend()
plt.savefig('plots2/posterior.jpg')
plt.close()
# %%
