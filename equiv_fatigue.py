#%%
import pymc3 as pm
import arviz as az
import pandas as pd 
import numpy as np

from scipy import stats

from typing import Dict
import theano.tensor as tt
import matplotlib.pyplot as plt

def basquin_rel(N, B,b):
    return B*(N**b)

B = 8e5
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
plt.savefig('plots/Guedes_model1.png')


fig = plt.figure()
ax = fig.gca()
ax.plot(logN, logSigma, label = 'mean')
ax.scatter(logN, logSrand, label = 'obs', color = 'r')
plt.legend()
plt.savefig('plots/Guedes_model2.png')

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

N_eq = 10e6
b = -0.1
#%%
frequency = np.array(data['Frequency [Hz]'].values)
amplitudes = np.array(list(data.columns)[1:], np.float64)
equivalent_cycles = np.fromiter(map(lambda x: equivalent_fatigue(x, data=data, amplitudes=amplitudes), frequency),
                                dtype=np.float64)
    
equivalent_cycles = equivalent_cycles ** b
frequency_density = np.fromiter(map(lambda x: get_freq_density(x, data=data), frequency), dtype=np.float64)
frequency_density /= frequency_density.max()
# %%

frequency = np.array(data['Frequency [Hz]'].values, dtype=np.float64).reshape(-1,1 )
frequency /= frequency.mean()
_,ax = plt.subplots(2,1, sharex=False, figsize=(12,5))
ax[0].hist(equivalent_cycles)
ax[1].plot(frequency, frequency_density)
ax[0].set_xlabel('Equivalent Cycles')
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Histogram')
plt.savefig('plots/equivalent_cycles.png')

#%%
#%%

with pm.Model() as GUEDE_disp_model:

    alpha = pm.HalfNormal('Alpha', sigma= 1.)
    beta = pm.HalfNormal('Beta', sigma= 1.)
    equivalent_fatigue_cycles = pm.Weibull('equivalent_fatigue', alpha=alpha, beta=beta, observed = equivalent_cycles)
    
    # beta = 1
    # scalling = pm.HalfNormal('Scale Factor', sigma= 2., shape=1)
    # scalling = 1
        
    # alpha = pm.Beta('alpha', alpha=2, beta=2, shape=1)
    # beta = pm.Beta('beta', alpha=2, beta=2, shape=1)
    nu = pm.HalfNormal('Nu', sigma=1)

    a = pm.Normal('a', mu=0, sigma = 10)
    A = pm.HalfCauchy('A', beta = 8)
    variation_coeff = pm.HalfCauchy('variation_coeff',beta=5)
    mean = a*logN + A
    
    noise_GUEDE = variation_coeff*mean
    likelihood = pm.StudentT('y', nu=nu, mu = mean, sigma = noise_GUEDE, observed = logSrand)
     
    # trace:Dict[str,np.ndarray] = pm.sample_smc()
    trace:Dict[str,np.ndarray] = pm.sample(draws=5000, chains = 3, tune=4000, target_accept=0.8)
    
    print(az.summary(trace))
    az.plot_trace(trace)
    plt.savefig('plots/trace_plot.png')

# %%
def Stress_stRenght(B, b, pf_slope, equivalent_cycles, N_eq = N_eq):
    stresses = pf_slope * equivalent_cycles    
    eq_stress = B * N_eq ** b
    G_RS = eq_stress - stresses
    return G_RS
    

label = 'damage'
with GUEDE_disp_model:
    B = tt.exp(A)    
    pf_slope = pm.Normal('pf_slope', mu = 5.5e5, sigma = 2000)
    damage = pm.Deterministic(label, 
                              Stress_stRenght(B, a, pf_slope, equivalent_cycles=equivalent_fatigue_cycles))
    samples:Dict[str,np.ndarray] = pm.sample_posterior_predictive(trace, samples = 150000, var_names=[label])

with GUEDE_disp_model:
    samples_2 = pm.sample_posterior_predictive(trace, samples = 2000, var_names=['equivalent_fatigue'])
# %%
_,ax = plt.subplots()
# for sample in samples[label]:
#     ax.hist(sample,bins = 40,density=True)

ax.hist(samples[label].flatten(),bins = 40,density=True)
ax.vlines(x=0, ymin=0, ymax=1e-4, colors='r', linestyles='dashed')

ax.set_xlabel(r'$G(R,S)=R-S$')
ax.set_ylabel('PDF')

plt.savefig('plots/output.png')

# %%
def indicator(x:np.ndarray):
    slice_ = x[x<0]
    return len(slice_.flatten())

print('SHAPE OF SAMPLES....',samples[label].shape)

N_samples = len(samples[label].flatten())
p_failure = indicator(samples[label].flatten())/N_samples

sigma_p_failure = p_failure*(1-p_failure)/N_samples

CoV = np.sqrt(sigma_p_failure/p_failure)

aux = {'p_Failure':p_failure, 'sigma_p_failure':sigma_p_failure, 'CoV':CoV}

print(aux)

p_failure = np.array([indicator(sample)/len(sample) for sample in samples[label]])
p_failure = np.fromiter(map(lambda x: indicator(x)/len(x), samples[label]), dtype=np.float64)

_,ax=plt.subplots()

ax.hist(p_failure, bins = 20, density=True)
ax.set_ylabel('Probabilidad de falla')
ax.set_xbound
plt.savefig('plots/p_failre.png')
#%%
_,ax = plt.subplots()
for sample in samples_2['equivalent_fatigue']:
    ax.hist(sample,bins = 40,density=True)

ax.hist(equivalent_cycles, density=True, label = 'Observed Data')
ax.set_xlabel(r'$G(R,S)=R-S$')
ax.set_ylabel('PDF')
plt.legend()
plt.savefig('plots/equiv_fatigue_posterior.png')
