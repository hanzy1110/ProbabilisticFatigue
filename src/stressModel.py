import os
import seaborn as sns
import pymc as pm
import nutpie
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from dataclasses import dataclass

# plt.style.use(['science', 'ieee'])


@dataclass
class CableProps:
    Eal: float
    Ecore: float
    rw: float
    rc: float
    layers: List[int]
    T: float

    def getEImin(self):
        I_a = np.pi * self.rw**4 / 64
        I_c = np.pi * self.rc**4 / 64
        layers = np.array(self.layers)
        return self.Eal * layers[0] * I_a + self.Ecore * layers[1] * I_c

    def getEIminEval(self, Eal, Ecore):
        I_a = np.pi * self.rw**4 / 64
        I_c = np.pi * self.rc**4 / 64
        layers = np.array(self.layers)
        return Eal * layers[0] * I_a + Ecore * layers[1] * I_c

    def getLambda(self, T):
        self.lambda_ = np.sqrt(T / self.getEImin())

    def PFSlope(self, a: float = 89.0):
        return (
            self.Eal
            * (2 * self.rw)
            * self.lambda_**2
            / (4 * (np.exp(-self.lambda_ * a) - 1 + self.lambda_ * a))
        )

    def PFSlopeEval(self, Eal, EIMin, T, a: float = 89.0):
        # lambda_ = np.sqrt(T/self.getEIminEval(Eal, Ecore))
        lambda_ = np.sqrt(T / EIMin)
        return (
            Eal
            * (2 * self.rw)
            * lambda_**2
            / (4 * (np.exp(-lambda_ * a) - 1 + lambda_ * a))
        )


class PFSlopeModel:
    def __init__(self, ydata, cableParams, Tpercentage, resultPath):
        df = pd.read_csv(ydata)
        self.ydata = df[df["tension"] == Tpercentage]
        self.cable = CableProps(**cableParams)
        self.results_folder = resultPath

    def restoreTrace(self):
        path = self.results_folder / "PF_TRACE.nc"
        if os.path.exists(path):
            self.trace = az.from_netcdf(filename=path)
        else:
            self.sampleModel(2000)

    def build_model(self):
        with pm.Model() as self.model:
            Eal = pm.Normal("Eal", mu=69e3, sigma=4e3)
            # Ecore = pm.Normal('Ecore', mu = 207e3, sd=4e3)
            EIMin = pm.Uniform("EIMin", lower=1e-4, upper=1)
            T = pm.Normal("T", mu=self.cable.T, sigma=self.cable.T / 40)
            pfSlope = pm.Deterministic(
                "pfSlope", self.cable.PFSlopeEval(Eal, 1e8 * EIMin, T)
            )
            noise = pm.Gamma("sigma", alpha=1, beta=2)
            likelihood = pm.Normal(
                "likelihood", mu=pfSlope, sigma=noise, observed=self.ydata
            )

    def sampleModel(self, ndraws):
        self.build_model()

        compiled_model = nutpie.compile_pymc_model(self.model)
        self.trace = nutpie.sample(compiled_model, draws=ndraws, tune=200, chains=3)
        az.to_netcdf(data=self.trace, filename=self.results_folder / "PF_TRACE.nc")

        with self.model:
            # self.trace = pm.sample(draws = ndraws, tune=2000, return_inferencedata=True)
            summ = az.summary(self.trace)
            az.plot_trace(self.trace)
            plt.savefig(self.results_folder / "PF_TRACEPLOT.jpg")
            az.plot_posterior(self.trace)
            self.postSamples = pm.sample_posterior_predictive(
                self.trace, var_names=["pfSlope"]
            )

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(3.3, 3.3)

        self.postSamples = self.postSamples.posterior_predictive

        ax.hist(self.postSamples["pfSlope"], density=True, label="Infered Data")
        sns.kdeplot(self.postSamples["pfSlope"], ax=ax)
        plt.hist(self.ydata, density=True, label="Experimental Data")
        ax.legend()
        ax.set_xlabel(r"\text{Poffemberger \& Swart Slope}")
        plt.savefig(self.results_folder / "PF_KDEPLOT.jpg")

        print(summ)
        az.to_netcdf(
            data=self.trace, filename=os.path.join(self.results_folder, "PFtrace.nc")
        )
