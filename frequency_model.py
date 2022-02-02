#%%
import pymc3 as pm
import arviz as az
import pandas as pd 
import numpy as np

from scipy import stats

from typing import Dict
import theano.tensor as tt
import matplotlib.pyplot as plt

def mixture_density(w, alpha, beta, x):
    logp = tt.log(w) + pm.Weibull.dist(alpha=alpha,beta=beta).logp(x)
    return tt.sum(tt.exp(logp))

# def mixture_density(alpha, beta, x, scalling=1):
#     logp = pm.Weibull.dist(alpha,beta).logp(x)
#     return scalling * tt.exp(logp)

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
frequency_density /= frequency_density.sum()
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
    
    mu_param_1 = pm.HalfNormal('Mu_param_1', sigma= 1) 
    mu_param_2 = pm.HalfNormal('Mu_param_2', sigma= 1)
    
    mu_sigma_1 = pm.HalfNormal('mu_sigma_1', sigma= 1)
    mu_sigma_2 = pm.HalfNormal('mu_sigma_2', sigma= 1)
    
    sigma_param_1 = pm.HalfNormal('sigma_param_1', sigma= 1) 
    sigma_param_2 = pm.HalfNormal('sigma_param_2', sigma= 1) 
    
    mu_1 = pm.Normal('Mu_1',mu = mu_param_1, sigma= 0.1)
    sigma_1 = pm.HalfNormal('Sigma_1', sigma= sigma_param_1)
    
    mu_2 = pm.Normal('Mu_2',mu =mu_param_2, sigma= 0.1)
    sigma_2 = pm.HalfNormal('Sigma_2', sigma=sigma_param_2)

    mu = tt.stack([mu_1, mu_2])
    sigma = tt.stack([sigma_1, sigma_2])
    
    w = pm.Dirichlet('w', a=np.array([1.,1.]))
    
    mix_1 = mixture_density(w, mu, sigma, x = frequency)
     
    noise_1 = pm.HalfNormal('noise_1', sigma = .1)     
    noise_2 = pm.HalfNormal('noise_2', sigma = .1)     
    
    N_1 = pm.StudentT('N_1', mu = mix_1, nu = noise_1, sigma=noise_2, observed = frequency_density)
    # N_1 = pm.Normal('N_1', mu = mix_1, sigma=noise_1, observed = frequency)
    # noise = 0.01   
    trace = pm.sample(draws=6000,
                      chains=3,
                      tune=6000,
                      target_accept=0.92,
                      return_inferencedata=False
                      )
    
    print(az.summary(trace))
    az.plot_trace(trace)

    
# %%
