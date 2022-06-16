#%%
import os
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

def get_freq_density(freq, data):
    cycles = np.array(data.loc[data['Frequency [Hz]'] == freq], dtype=np.float64)
    return cycles.sum()

class FatigueModel:
    def __init__(self, resultsFolder:str, b_mean:float, observedDataPath:str):

        if not os.path.exists(resultsFolder):
            os.makedirs(resultsFolder)

        self.resultsFolder = resultsFolder
        data = pd.read_csv(observedDataPath)
        data.set_index(data['Frequency [Hz]'], inplace=True)
        data.drop(data.index.values[-1], inplace=True)

        self.b = b_mean
        self.freq_prev = np.array(data['Frequency [Hz]'].values)
        self.amplitudes = np.array(list(data.columns)[1:], np.float64)
        self.frequency_density = np.fromiter(map(lambda x: get_freq_density(x, data=data), self.freq_prev), dtype=np.float64)
        freq = [int(d) for d in data['Frequency [Hz]'].values]
        self.frequency = np.array(freq)
        self.frequency = hist_sample([self.frequency_density, self.frequency], n=2500)

    def NormalizeData(self, plotExp:bool=True):

        # frequency = np.array(data['Frequency [Hz]'].values, dtype=np.float64).reshape(-1,1 )
        _,ax = plt.subplots(1,1, sharex=False, figsize=(10,8))
        ax.hist(self.freq_prev)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Density')
        plt.savefig(os.path.join(self.resultsFolder, 'expData.jpg'))
        plt.close()
        self.maxFreq = self.frequency.max()
        self.frequency = np.array(self.frequency, dtype=np.float64)/self.maxFreq

    def buildMixtureFreqModel(self,):

        self.NormalizeData()
        with pm.Model() as self.frequency_model:
            w = pm.Dirichlet('w', np.ones(2))
            # w = np.array([1,0])
            alphas = pm.HalfNormal('alphas', sigma=10, shape=(2,))
            betas = pm.HalfNormal('betas', sigma=10, shape=(2,))

            weib_0 = pm.Weibull.dist(alpha=alphas[0], beta=betas[0])
            weib_1 = pm.Weibull.dist(alpha=alphas[1], beta=betas[1])
            GMM = pm.Mixture('likelihood',
                             w=w,comp_dists=[weib_0, weib_1],
                             observed=self.frequency)

    def sampleModel(self, ndraws):

        with self.frequency_model:

            self.trace = pm.sample_smc(draws=2000,chains=2,parallel=False)
            print(az.summary(self.trace))
            az.plot_trace(self.trace)
            plt.savefig(os.path.join(self.resultsFolder, 'traceplot.jpg'))
            plt.close()
            az.plot_posterior(self.trace)

            plt.savefig(os.path.join(self.resultsFolder, 'posteriorArviz.jpg'))
            plt.close()

    def samplePosterior(self):

        with self.frequency_model:
            samples = pm.sample_posterior_predictive(self.trace, var_names=['likelihood'])

        _,ax = plt.subplots(1,1, sharex=False, figsize=(12,5))
        sns.kdeplot(self.frequency, ax=ax, label='observed')
        sns.kdeplot(samples['likelihood'].flatten(), ax=ax, label='Inference')
        plt.legend()
        plt.savefig(os.path.join(self.resultsFolder, 'posterior.jpg'))
        plt.close()
