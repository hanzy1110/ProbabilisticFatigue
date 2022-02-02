#%%
import pymc3 as pm
import arviz as az
import pandas as pd 
import numpy as np

from scipy import stats

from typing import Dict
import theano.tensor as tt
import matplotlib.pyplot as plt

def from_posterior(samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return x,y

# def mixture_density(w, alpha, beta, scalling, x):
#     logp = tt.log(w) + pm.Weibull.dist(alpha,beta).logp(x)
#     return scalling * tt.sum(tt.exp(logp))

def mixture_density(alpha, beta, x, scalling=1):
    logp = pm.Weibull.dist(alpha,beta).logp(x)
    return scalling * tt.exp(logp)

def equivalent_fatigue(freq, data, amplitudes):
    cycles = np.array(data.loc[data['Frequency [Hz]'] == freq], dtype=np.int64)[0][1:]
    basquin_stress = amplitudes**b
    
    return np.dot(cycles, basquin_stress)

def get_freq_density(freq, data):
    cycles = np.array(data.loc[data['Frequency [Hz]'] == freq], dtype=np.int64)
    return cycles.sum()
    
data = pd.read_csv('cleansed_csvs/5BBOL2-137_VANO 136_OPGW_807625_19_02_2020.csv')
data.set_index(data['Frequency [Hz]'])
data.drop(data.index.values[-1], inplace=True)

N_eq = 1e7
b = -0.1
#%%
frequency = np.array(data['Frequency [Hz]'].values)
amplitudes = np.array(list(data.columns)[1:], np.float64)
equivalent_cycles = np.fromiter(map(lambda x: equivalent_fatigue(x, data=data, amplitudes=amplitudes), frequency),
                                dtype=np.float64)
    
equivalent_cycles = (equivalent_cycles/N_eq) ** b
frequency_density = np.fromiter(map(lambda x: get_freq_density(x, data=data), frequency), dtype=np.float64)
frequency_density /= frequency_density.max()
# %%

frequency = np.array(data['Frequency [Hz]'].values, dtype=np.float64).reshape(-1,1 )
frequency /= frequency.mean()
_,ax = plt.subplots(2,1, sharex=False, figsize=(12,5))
ax[0].hist(equivalent_cycles)
ax[1].plot(frequency, frequency_density)
# %%

with pm.Model() as frequency_model:

    # scalling = pm.HalfNormal('Scale Factor', sigma= 2., shape=1)
    # scalling = 1
        
    alpha_1 = pm.HalfNormal('Alpha_1', sigma= 1.)
    beta_1 = pm.HalfNormal('Beta_1', sigma= 1., )
    
    alpha_2 = pm.HalfNormal('Alpha_2', sigma= 1.)
    beta_2 = pm.HalfNormal('Beta_2', sigma= 1., )
    
    mix_1 = mixture_density(alpha_1, beta_1, x = frequency)
    mix_2 = mixture_density(alpha_2, beta_2, x = frequency)
    
    noise_1 = pm.HalfNormal('noise_1', sigma = 1)
    noise_2 = pm.HalfNormal('noise_2', sigma = 1)
        
    N_1 = pm.StudentT('N_1', mu = mix_1, nu=noise_1)
    N_2 = pm.StudentT('N_2', mu = mix_2, nu=noise_2)
    
    w = pm.Dirichlet('w', a=np.array([1.,1.]))
    
    likelihood = pm.Mixture('Likelihood',w=w, comp_dists=[N_1, N_2], observed=frequency_density)
    
    # noise = 0.01
       
    trace = pm.sample(draws=3000, chains = 4, tune=2000, target_accept=0.8)
    
    print(az.summary(trace))
    az.plot_trace(trace)

    