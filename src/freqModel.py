from functools import reduce
import os
import pathlib
from pandas.tseries import frequencies
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


def parse_exp_data_freq(data_path: pathlib.Path):
    if "csv" in data_path.suffix:
        data = pd.read_csv(data_path)
        data.set_index(data["Frequency [Hz]"], inplace=True)
        data.drop(data.index.values[-1], inplace=True)
        freq_prev = np.array(data["Frequency [Hz]"].values)
        frequency_density = np.fromiter(
            map(lambda x: get_freq_density(x, data=data), freq_prev), dtype=np.float64
        )

        print(f"freq_prev: {freq_prev}")
        print(f"frequency_density: {frequency_density}")
        freq_prev = np.array(freq_prev, dtype=np.float64)

    else:
        data = pd.read_excel(data_path, sheet_name=None)
        data = list(map(get_cycles_freqs, data.values()))
        data_joined = reduce(lambda x, y: join_hists(x, y, "frequencies"), data, None)

        frequency_density, freq_prev = (
            data_joined["cycles"] * 1e6,
            data_joined["frequencies"],
        )

    return frequency_density, freq_prev


def get_cycles_freqs(df: pd.DataFrame):
    cycles = np.array(df[df.columns[-1]].values[:-1], dtype=np.float64)
    frequencies = np.array(df[df.columns[0]].values[:-1], np.float64)
    return {"cycles": cycles, "frequencies": frequencies}


def get_cycles_amps(df: pd.DataFrame):
    cycles = np.array(df.iloc[-1].values[1:-1], dtype=np.float64)
    amplitudes = np.array(list(df.columns)[1:-1], np.float64)

    return {"cycles": cycles, "amplitudes": amplitudes}


def join_hists(new, acc, label="amplitudes"):
    if new is not None:
        amplitudes = acc[label]
        cycles = acc["cycles"]
        for i, c in enumerate(new["cycles"]):
            mask = np.isclose(cycles, c)
            if any(mask):
                amplitudes[mask] += new[label][i]
            else:
                cycles = np.hstack([cycles, c])
                amplitudes = np.hstack([amplitudes, new[label][i]])

        sorted_idxs = np.argsort(cycles)
        return {"cycles": np.sort(cycles), label: amplitudes[sorted_idxs]}
    return acc


def hist_sample(hist, n):
    """Genertae histogram sample
    Args:
        hist (array): hist[0]: frecuencies/probs of X values, hist[1]: X values
        n ([type]): number of samples
    Returns:
        [list]: list with samples
    """
    ps = np.array(hist[0] / hist[0].sum())
    assert np.isclose(ps.sum(), 1)
    return np.random.choice(hist[1], size=n, p=ps)


def get_freq_density(freq, data):
    cycles = np.array(data.loc[data["Frequency [Hz]"] == freq], dtype=np.float64)
    return cycles.sum()


def calculate_freq_dist(freq_data: pathlib.Path, plot=False):
    # data = pd.read_csv(freq_data)
    frequency_density, freq_prev = parse_exp_data_freq(freq_data)
    frequency = hist_sample([frequency_density, freq_prev], n=10000)

    return frequency


def total_cycles_per_year(
    cycling_hours, n_years, freq_data, ndraws=1, ls=1.1, tau=1e-4
):
    freq = calculate_freq_dist(freq_data)
    print(f"frequency => {freq}")
    n_mean = cycling_hours * freq.mean() * 3600
    # return n_mean * np.ones_like(np.arange(n_years))
    # Use the follwing when modelling random amounts:
    cov = (n_mean / tau) * pm.gp.cov.Matern52(1, ls)
    X = np.linspace(0, n_years, n_years)[:, None]
    K = cov(X).eval()
    mu = n_mean * np.ones(len(K))
    cycles = pm.draw(
        pm.MvNormal.dist(mu=mu, cov=K, shape=len(K)), draws=ndraws, random_seed=rng
    ).T

    return cycles


class LoadModel:
    def __init__(
        self,
        resultsFolder: pathlib.Path,
        observedDataPath: pathlib.Path,
        ydata: pathlib.Path,
        cableProps: Dict,
        Tpercentage: int,
    ):
        if not os.path.exists(resultsFolder):
            os.makedirs(resultsFolder)

        self.results_folder = resultsFolder
        # data = pd.read_csv(observedDataPath)
        # data.set_index(data['Frequency [Hz]'], inplace=True)
        # data.drop(data.index.values[-1], inplace=True)

        df = pd.read_csv(ydata)
        self.ydata = df[df["tension"] == Tpercentage]
        self.cable = CableProps(**cableProps)

        cycles, amplitudes = self.parse_exp_data(observedDataPath)
        self.amplitudes_sample = hist_sample([cycles, amplitudes], n=2500)

    def parse_exp_data(self, data_path: pathlib.Path):
        if "csv" in data_path.suffix:
            data = pd.read_csv(data_path)
            cycles = np.array(data.iloc[-1].values[1:], dtype=np.float64)
            amplitudes = np.array(list(data.columns)[1:], np.float64)
        else:
            data = pd.read_excel(data_path, sheet_name=None)
            data = list(map(get_cycles_amps, data.values()))
            data_joined = reduce(join_hists, data, None)

            cycles, amplitudes = (
                data_joined["cycles"] * 1e6,
                data_joined["amplitudes"] * 25400,
            )
        return cycles, amplitudes

    def NormalizeData(self, plotExp: bool = True):
        # frequency = np.array(data['Frequency [Hz]'].values, dtype=np.float64).reshape(-1,1 )
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=(10, 8))
        fig.set_size_inches(3.3, 3.3)
        ax.hist(self.amplitudes_sample)
        ax.set_xlabel("Amplitudes")
        ax.set_ylabel("Frecuencia")
        plt.savefig(self.results_folder / "LOAD_EXP_DATA.png", dpi=600)
        plt.close()
        self.maxAmp = self.amplitudes_sample.max()
        self.amplitudes_sample = (
            np.array(self.amplitudes_sample, dtype=np.float64) / self.maxAmp
        )

    def buildMixtureFreqModel(
        self,
    ):
        self.NormalizeData()
        with pm.Model() as self.amplitude_model:
            alpha = pm.HalfNormal("alpha", sigma=10)
            beta = pm.HalfNormal("beta", sigma=10)
            self.loads = pm.Weibull(
                "likelihood", alpha=alpha, beta=beta, observed=self.amplitudes_sample
            )

            Eal = pm.Normal("Eal", mu=69e3, sigma=4e3)
            # Ecore = pm.Normal('Ecore', mu = 207e3, sd=4e3)
            EIMin = pm.Uniform("EIMin", lower=1e-4, upper=1)
            T = pm.Normal("T", mu=self.cable.T, sigma=self.cable.T / 40)
            self.pfSlope = pm.Deterministic(
                "pfSlope", self.cable.PFSlopeEval(Eal, 1e8 * EIMin, T)
            )
            noise = pm.Gamma("sigma", alpha=1, beta=2)
            likelihood = pm.Normal(
                "likelihood2", mu=self.pfSlope, sigma=noise, observed=self.ydata
            )

    def sampleModel(self, ndraws):
        print("Sampling Load Model")
        compiled_model = nutpie.compile_pymc_model(self.amplitude_model)
        self.trace = nutpie.sample(compiled_model, draws=ndraws, tune=1000, chains=4)
        az.to_netcdf(
            data=self.trace, filename=self.results_folder / "AMPLITUDE_TRACE.nc"
        )

        with self.amplitude_model:
            print(az.summary(self.trace))
            az.plot_trace(self.trace)
            plt.savefig(self.results_folder / "AMPLITUDE_TRACEPLOT.jpg")
            plt.close()
            az.plot_posterior(self.trace)
            plt.savefig(self.results_folder / "AMPLITUDE_POSTERIOR.jpg")
            plt.close()

    def samplePosterior(self):
        with self.amplitude_model:
            samples = pm.sample_posterior_predictive(
                self.trace, var_names=["likelihood"]
            )

        fig, ax = plt.subplots(1, 1, sharex=False, figsize=(12, 5))
        fig.set_size_inches(3.3, 3.3)
        ax.hist(self.amplitudes_sample, density=True)
        sns.kdeplot(samples["likelihood"].flatten(), ax=ax, label="Inferencia")

        plt.legend()
        plt.savefig(self.results_folder / "posterior.jpg", dpi=600)
        plt.close()

    def restoreTrace(self):
        path = self.results_folder / "AMPLITUDE_TRACE.nc"

        if os.path.exists(path):
            self.trace = az.from_netcdf(filename=path)
        else:
            self.sampleModel(2000)
        return self.trace
