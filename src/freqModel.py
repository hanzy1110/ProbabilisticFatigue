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

class LoadModel:
    def __init__(self, resultsFolder:str, observedDataPath:str):

        if not os.path.exists(resultsFolder):
            os.makedirs(resultsFolder)

        self.resultsFolder = resultsFolder
        # data = pd.read_csv(observedDataPath)
        # data.set_index(data['Frequency [Hz]'], inplace=True)
        # data.drop(data.index.values[-1], inplace=True)

        data = pd.read_csv(observedDataPath)
        cycles = np.array(data.iloc[-1].values[1:], dtype=np.float64)
        amplitudes = np.array(list(data.columns)[1:], np.float64)
        self.amplitudes_sample = hist_sample([cycles, amplitudes], n=2500)

    def NormalizeData(self, plotExp:bool=True):

        # frequency = np.array(data['Frequency [Hz]'].values, dtype=np.float64).reshape(-1,1 )
        _,ax = plt.subplots(1,1, sharex=False, figsize=(10,8))
        ax.hist(self.amplitudes_sample)
        ax.set_xlabel('Amplitudes')
        ax.set_ylabel('Density')
        plt.savefig(os.path.join(self.resultsFolder, 'expData.jpg'))
        plt.close()
        self.maxAmp = self.amplitudes_sample.max()
        self.amplitudes_sample = np.array(self.amplitudes_sample, dtype=np.float64)/self.maxAmp

    def buildMixtureFreqModel(self,):

        self.NormalizeData()
        with pm.Model() as self.amplitude_model:
            alpha = pm.HalfNormal('alpha', sigma=10)
            beta = pm.HalfNormal('beta', sigma=10)
            self.loads = pm.Weibull('likelihood',alpha=alpha, beta=beta,
                                    observed=self.amplitudes_sample)
    def sampleModel(self, ndraws):

        with self.amplitude_model:

            self.trace = pm.sample(draws=ndraws,
                              chains=3,
                              tune=3000,
                              # target_accept=0.97,
                              return_inferencedata=True
                              )


            print(az.summary(self.trace))
            az.plot_trace(self.trace)
            plt.savefig(os.path.join(self.resultsFolder, 'traceplot.jpg'))
            plt.close()
            az.plot_posterior(self.trace)

            az.to_netcdf(self.trace,
                         filename=os.path.join(self.resultsFolder, 'trace.nc'))

            # df = pm.backends.tracetab.trace_to_dataframe(self.trace)
            # df.to_csv(os.path.join(self.resultsFolder, 'trace.csv'))
            plt.savefig(os.path.join(self.resultsFolder, 'posteriorArviz.jpg'))
            plt.close()

    def samplePosterior(self):

        with self.amplitude_model:
            samples = pm.sample_posterior_predictive(self.trace, var_names=['likelihood'])

        _,ax = plt.subplots(1,1, sharex=False, figsize=(12,5))
        ax.hist(self.amplitudes_sample, density=True)
        sns.kdeplot(samples['likelihood'].flatten(), ax=ax, label='Inference')

        plt.legend()
        plt.savefig(os.path.join(self.resultsFolder, 'posterior.jpg'))
        plt.close()

    def restoreTrace(self):
        try:
            # trace = pd.read_csv(os.path.join(self.resultsFolder, 'trace.csv'))
            self.trace = az.from_netcdf(filename=os.path.join(self.resultsFolder, 'trace.nc'))
        except Exception as e:
            print(e)
            self.sampleModel(2000)

        return self.trace
