import pymc3 as pm
import arviz as az
import pandas as pd 
import numpy as np

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
variation_coeff = 0.2

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

mean_amp = amplitudes.mean()
amplitudes /= mean_amp
tot_cycles /= np.sum(tot_cycles)

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


def Miner_sRule(B,b, amplitudes, alpha, beta, scalling, max_cycles = 1e7):

    total_cycles = mixture_density(alpha, beta, scalling, x=amplitudes) * max_cycles
    
    stresses = 4e1* amplitudes
    
    damage = total_cycles * (stresses/B) ** (1/b)
    return damage.sum()


