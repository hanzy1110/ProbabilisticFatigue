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

def basquin_rel(N, B,b):
    return B*(N**b)

B = 10e5
b = -1e-1

Ns = np.linspace(1e4, 1e7, 200)
sigmas = np.array([basquin_rel(val, B,b) for val in Ns])

logN = np.log(Ns)
logSigma = np.log(sigmas)
variation_coeff = 1

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

amplitudes /= amplitudes.mean()
# tot_cycles /= np.sum(tot_cycles)
tot_cycles /= tot_cycles.max()

# plt.plot(amplitudes[1:], tot_cycles)

plt.hist(tot_cycles, bins = amplitudes)

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

with pm.Model() as GUEDE_disp_model:

    alpha = pm.HalfNormal('Alpha', sigma= 1., shape=1)
    beta = pm.HalfNormal('Beta', sigma= 1., shape=1)
    # beta = 1
    scalling = pm.HalfNormal('Scale Factor', sigma= 2., shape=1)
    # scalling = 1
        
    # alpha = pm.Beta('alpha', alpha=2, beta=2, shape=1)
    # beta = pm.Beta('beta', alpha=2, beta=2, shape=1)
    noise = pm.HalfNormal('Noise', sigma=1)

    a = pm.Normal('a', mu=0, sigma = 10)
    A = pm.HalfCauchy('A', beta = 8)
    variation_coeff = pm.HalfCauchy('variation_coeff',beta=5)
    mean = a*logN + A
    noise_GUEDE = variation_coeff*mean
    
    normed_disp = pm.Normal('obs', 
                            mixture_density(alpha, beta, scalling, amplitudes),
                            noise, 
                            observed=tot_cycles)
    
    likelihood = pm.Normal('y', mu = mean, sigma = noise_GUEDE, observed = logSrand)
 
    # trace:Dict[str,np.ndarray] = pm.sample_smc()
    trace:Dict[str,np.ndarray] = pm.sample(draws=4000, chains = 4, tune=2000, target_accept=0.92)
    
    print(az.summary(trace))
    az.plot_trace(trace)

# %%
def weibull_samples(a, b, scale = 1, size=None):
    uniform = np.random.uniform(size=size)
    
    return b * (-np.log(uniform/scale)) ** (1 / a)

def theano_weibull_samples(a, b, scale = 1, size=None):
    uniform = np.random.uniform(size=size)
    
    return b * (-tt.log(uniform/scale)) ** (1 / a)

def Miner_sRule(B,b, n_samples, alpha, beta, scalling):

    samples = theano_weibull_samples(a=alpha, b=beta, size=n_samples)
    total_cycles = mixture_density(alpha, beta, scalling, x=samples) * n_samples
    
    stresses = 3e9* samples
    
    baskin = (stresses/B) ** (1/b)
    
    damage = tt.dot(total_cycles, baskin)
    return damage
    

label = 'damage5'
with GUEDE_disp_model:
    B = tt.exp(A)
    damage = pm.Deterministic(label, 
                              Miner_sRule(B, a, 200000, alpha, beta, scalling))
    samples:Dict[str,np.ndarray] = pm.sample_posterior_predictive(trace, samples = 10000, var_names=[label])

# %%
plt.hist(samples[label],bins = 40,density=True)
# %%
def indicator(x:np.ndarray):
    slice_ = x[x<0]
    return np.ones_like(slice_).sum()

perf = 1-samples[label]

p_failure = indicator(perf)/len(perf)
print('p_failure--->', p_failure)


# %%
