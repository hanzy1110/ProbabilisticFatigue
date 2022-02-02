#%%
import pymc3 as pm
import arviz as az
import pandas as pd 
import numpy as np
from pymc3.gp.util import plot_gp_dist

from scipy import stats

from typing import Dict
import theano.tensor as tt
import matplotlib.pyplot as plt
    
def mixture_density(alpha, beta, scalling, x):
    logp = pm.Weibull.dist(alpha, beta).logp(x)
    return scalling * tt.exp(logp)
    

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

mean_amp = amplitudes.mean()
amplitudes /= mean_amp

total_cycles = np.sum(tot_cycles)
tot_cycles /= total_cycles

total_cycles = 100000

plt.plot(amplitudes[1:], tot_cycles)

#%%
# amplitudes = amplitudes[1:].reshape(-1,1)
amplitudes = amplitudes[1:]

alpha_0 = np.array([1.5])
beta_0 = np.array([1.])
scalling =np.array([1])

yhat = mixture_density(alpha_0, beta_0, scalling, amplitudes).eval()
y = yhat + np.random.normal(loc=0, scale=.1, size=len(yhat))

plt.plot(amplitudes, yhat, label='Approximación')
plt.scatter(amplitudes, y, label = 'Approx + Ruido')
plt.scatter(amplitudes,tot_cycles, label = 'Datos Observados')
plt.legend()

#%%

with pm.Model() as disp_model:

    alpha = pm.HalfNormal('Alpha', sigma= 1., shape=1)
    beta = pm.HalfNormal('Beta', sigma= 1., shape=1)
    # beta = 1
    scalling = pm.HalfNormal('Scale Factor', sigma= 2., shape=1)
    # scalling = 1
        
    # alpha = pm.Beta('alpha', alpha=2, beta=2, shape=1)
    # beta = pm.Beta('beta', alpha=2, beta=2, shape=1)
    noise = pm.HalfNormal('Noise', sigma=1)
    
    normed_disp = pm.Normal('obs', 
                            mixture_density(alpha, beta, scalling, amplitudes),
                            noise, 
                            observed=tot_cycles)
    
    # trace:Dict[str,np.ndarray] = pm.sample_smc()
    trace:Dict[str,np.ndarray] = pm.sample(draws=4000, chains = 4, tune=2000, target_accept=0.92)
    
    print(az.summary(trace))
    az.plot_trace(trace)

# %%
new_amps = np.linspace(amplitudes.min(), amplitudes.max(), num=100)
with disp_model:
    mixture = pm.Deterministic('mixture',mixture_density(alpha, beta, scalling, new_amps))
    posterior_samples = pm.sample_posterior_predictive(trace, samples=24000, var_names=['mixture'])
#%%
_, ax = plt.subplots(figsize=(12,10))
plot_gp_dist(ax=ax, samples = posterior_samples['mixture'], x=new_amps)
ax.scatter(amplitudes, tot_cycles, label='Observed Data')
ax.set_title('Posterior Samples')
ax.set_xlabel(r'$\frac{Amplitud}{\mu_{Amp}}$')
ax.set_ylabel(r'$\frac{Frequencia}{\Sigma_{freq}}$')
plt.legend()


# %%

def weibull_samples(a, b, scale = 1, size=None):
    uniform = np.random.uniform(size=size)
    
    return b * (-np.log(uniform/scale)) ** (1 / a)

def theano_weibull_samples(a, b, scale = 1, size=None):
    uniform = np.random.uniform(size=size)
    
    return b * (-tt.log(uniform/scale)) ** (1 / a)


with disp_model:
    amp_samples = pm.Deterministic('amplitudes', 
                                   theano_weibull_samples(alpha, 
                                                          beta, 
                                                          scale=1, 
                                                          size=int(total_cycles)))
    
    samples:Dict[str,np.ndarray] = pm.sample_posterior_predictive(trace, samples = 1000, var_names=['amplitudes'])

#%%
mean_alpha = trace['Alpha'].mean()
mean_beta = trace['Beta'].mean()
mean_scalling = trace['Scale Factor'].mean()
mean_scalling = 1

_, ax = plt.subplots(figsize=(12,10))

sigma = np.array([sample.std() for sample in samples['amplitudes']]).mean()
print('sigma--->', sigma)

for sample in samples['amplitudes'][:10]:
    
    hist, bins = np.histogram(sample, bins = 24)
    total = hist.sum()
    
    y = mixture_density(mean_alpha, mean_beta, mean_scalling, bins).eval()
    ax.plot(bins, y * total)    
    # hist,bins = np.histogram(sample)
    ax.hist(sample)

# ax.scatter(amplitudes, tot_cycles, label='Datos Observados')
ax.set_title('Posterior Samples')
ax.set_xlabel(r'$\frac{Amplitude}{\mu_{Amp}}$')
ax.set_ylabel(r'$\frac{Frequency}{\Sigma_{freq}}$')
# plt.legend()

# %%





# %%
