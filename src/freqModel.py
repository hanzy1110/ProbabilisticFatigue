import os
import pathlib
import pymc as pm
import nutpie
import arviz as az
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

from typing import Dict
import matplotlib.pyplot as plt
from .stressModel import CableProps

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

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

def calculate_freq_dist(freq_data: pathlib.Path, plot=False):
    data = pd.read_csv(freq_data)
    # data['Frequency [Hz]'] = np.array(data['Frequency [Hz]'].values, dtype=np.int64)
    data.set_index(data['Frequency [Hz]'], inplace=True)
    data.drop(data.index.values[-1], inplace=True)

    freq_prev = np.array(data['Frequency [Hz]'].values)
    frequency_density = np.fromiter(map(lambda x: get_freq_density(x, data=data), freq_prev), dtype=np.float64)
    frequency = hist_sample([frequency_density, freq_prev], n=10000)

    return frequency

def total_cycles_per_year(cycling_hours, n_years, freq_data, ls=0.2, tau=2.0):

    mean_freq = calculate_freq_dist(freq_data).mean()
    n_mean = cycling_hours * mean_freq * 3600
    return n_mean * np.ones_like(np.arange(n_years))
    # Use the follwing when modelling random amounts:
    # cov = tau * pm.gp.cov.Matern52(1, ls)
    # X = np.linspace(0, n_years, n_years)[:, None]
    # K = cov(X).eval()
    # mu = n_mean * np.ones(len(K))
    # cycles = pm.draw(pm.MvNormal.dist(mu=mu, cov=K, shape=len(K)), draws=3, random_seed=rng).T


class LoadModel:
    def __init__(self, resultsFolder:pathlib.Path,
                 observedDataPath:pathlib.Path,
                 ydata:pathlib.Path,
                 cableProps:Dict, Tpercentage:int):

        if not os.path.exists(resultsFolder):
            os.makedirs(resultsFolder)

        self.results_folder = resultsFolder
        # data = pd.read_csv(observedDataPath)
        # data.set_index(data['Frequency [Hz]'], inplace=True)
        # data.drop(data.index.values[-1], inplace=True)
        
        df = pd.read_csv(ydata)
        self.ydata = df[df['tension']==Tpercentage] 
        self.cable = CableProps(**cableProps) 

        data = pd.read_csv(observedDataPath)
        cycles = np.array(data.iloc[-1].values[1:], dtype=np.float64)
        amplitudes = np.array(list(data.columns)[1:], np.float64)
        self.amplitudes_sample = hist_sample([cycles, amplitudes], n=2500)

    def NormalizeData(self, plotExp:bool=True):

        # frequency = np.array(data['Frequency [Hz]'].values, dtype=np.float64).reshape(-1,1 )
        fig, ax = plt.subplots(1,1, sharex=False, figsize=(10,8))
        fig.set_size_inches(3.3, 3.3)
        ax.hist(self.amplitudes_sample)
        ax.set_xlabel('Amplitudes')
        ax.set_ylabel('Density')
        plt.savefig(self.results_folder / 'LOAD_EXP_DATA.png', dpi=600)
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

            Eal = pm.Normal('Eal', mu = 69e3, sigma=4e3)
            # Ecore = pm.Normal('Ecore', mu = 207e3, sd=4e3)
            EIMin = pm.Uniform('EIMin', lower=1e-4, upper=1)
            T = pm.Normal('T', mu=self.cable.T, sigma=self.cable.T/40)
            self.pfSlope = pm.Deterministic('pfSlope', self.cable.PFSlopeEval(Eal, 1e8*EIMin, T))
            noise = pm.Gamma('sigma', alpha=1, beta=2)
            likelihood = pm.Normal('likelihood2', mu=self.pfSlope, sigma=noise, observed=self.ydata)

    def sampleModel(self, ndraws):

        compiled_model = nutpie.compile_pymc_model(self.amplitude_model)
        self.trace = nutpie.sample(compiled_model, draws=ndraws, tune=1000, chains=4)
        az.to_netcdf(data=trace, filename= self.results_folder / "AMPLITUDE_TRACE.nc")

        with self.amplitude_model:

            # self.trace = pm.sample(draws=ndraws,
            #                   chains=3,
            #                   tune=3000,
            #                   # target_accept=0.97,
            #                   return_inferencedata=True
            #                   )

            print(az.summary(self.trace))
            az.plot_trace(self.trace)
            plt.savefig(self.results_folder/ 'AMPLITUDE_TRACEPLOT.jpg')
            plt.close()
            az.plot_posterior(self.trace)
            plt.savefig(self.results_folder/ 'AMPLITUDE_POSTERIOR.jpg')
            plt.close()

    def samplePosterior(self):

        with self.amplitude_model:
            samples = pm.sample_posterior_predictive(self.trace, var_names=['likelihood'])

        fig, ax = plt.subplots(1,1, sharex=False, figsize=(12,5))
        fig.set_size_inches(3.3, 3.3)
        ax.hist(self.amplitudes_sample, density=True)
        sns.kdeplot(samples['likelihood'].flatten(), ax=ax, label='Inference')

        plt.legend()
        plt.savefig(self.results_folder/ 'posterior.jpg', dpi=600)
        plt.close()

    def restoreTrace(self):
        path = self.results_folder/ 'AMPLITUDE_TRACE.nc'

        if os.path.exists(path):
            self.trace = az.from_netcdf(filename=path)
        else:
            self.sampleModel(2000)
        return self.trace
