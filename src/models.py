#%%
import os
import numpy as np
import scipy
import pymc3 as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from pymc3.gp.util import plot_gp_dist
import json

from .plotUtils import plot_mean, plot_var, plot_total, get_ℓ_prior
import jax.numpy as jnp
from jax import vmap ,jit

def logistic(x, a, x0, c):
    # a is the slope, x0 is the location
    return c *(1 - pm.math.invlogit(a * (x - x0)))

def log1exp(arr:jnp.ndarray):
    return jnp.log(1+jnp.exp(arr))

class WohlerCurve:

    def __init__(self, resultsFolder:str,
                 observedDataPath:str='data/SN_curve.mat', )->None:

        if not os.path.exists(resultsFolder):
            os.mkdir(resultsFolder)

        self.resultsFolder = resultsFolder
        # Both are given as rows!
        SN_data = scipy.io.loadmat(observedDataPath)
        self.log_N = SN_data['X'].flatten()
        self.S = SN_data['Y'].flatten()

    def NormalizeData(self, plotExp:bool):

        self.NMax = np.log(self.log_N.max()*1e6)
        self.log_N = np.log(self.log_N*1e6)/self.NMax

        self.SMax = self.S.max()
        self.S/=self.SMax
        self.SNew = np.linspace(self.S.min(), self.S.max(), 100)[:, None]

        if plotExp:
            _,ax = plt.subplots(1,1, figsize=(12,8))
            ax.scatter(self.S,self.log_N)
            ax.set_ylabel('log(N)/log(NMax)')
            ax.set_xlabel('S/Smax')
            plt.savefig(os.path.join(self.resultsFolder,'experimental.jpg'))
            plt.close()

    def build_HeteroskedasticModel(self):


        self.NormalizeData(plotExp=True)
        S = self.S.reshape((-1,1))

        ℓ_μ, ℓ_σ = [stat for stat in get_ℓ_prior(self.S)]

        with pm.Model() as self.SNCurveModel:
            ℓ = pm.Gamma("ℓ", mu=ℓ_μ, sigma=ℓ_σ)
            η = pm.Gamma("η", alpha=2, beta=1)

            # x0 = pm.Gamma("x0", alpha=2, beta=1)
            # a = pm.Gamma("a", alpha=2, beta=1)
            # c = pm.Gamma("c", alpha=2, beta=1)

            # cov_base = η ** 2 * pm.gp.cov.Exponential(input_dim=1, ls=ℓ) + pm.gp.cov.WhiteNoise(sigma=1e-6)
            cov = η ** 2 * pm.gp.cov.Exponential(input_dim=1, ls=ℓ) + pm.gp.cov.WhiteNoise(sigma=1e-6)
            # cov = pm.gp.cov.ScaledCov(1, scaling_func=logistic, args=(a, x0, c), cov_func=cov_base)

            self.gp_ht = pm.gp.Latent(cov_func=cov)
            mu_f = self.gp_ht.prior("mu_f", X=S)
            μ_f  = pm.Deterministic('μ_f', pm.math.log1pexp(mu_f))
            σ_ℓ = pm.Gamma("σ_ℓ", mu=ℓ_μ, sigma=ℓ_σ)
            σ_η = pm.Gamma("σ_η", alpha=2, beta=1)

            σ_cov = σ_η ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=σ_ℓ) + pm.gp.cov.WhiteNoise(sigma=1e-6)

            self.σ_gp = pm.gp.Latent(cov_func=σ_cov)
            σ_f = self.σ_gp.prior("lg_σ_f", X=S)
            σ_f = pm.Deterministic("σ_f", pm.math.exp(σ_f))

            nu = pm.Gamma("nu", alpha=2, beta=1)
            lik_ht = pm.StudentT("lik_ht",nu=nu,  mu=μ_f, sigma=σ_f, observed=self.log_N)

    def sampleModel(self, ndraws):

        with self.SNCurveModel:
            self.trace = pm.sample_smc(draws=ndraws, parallel=True)
            # self.trace = pm.sample(draws=ndraws, chains=4, tune=2000, target_accept=0.97)
            if not isinstance(self.trace, az.InferenceData):
                self.trace = az.convert_to_inference_data(self.trace)

            # self.trace = pm.sample()
            summ = az.summary(self.trace)
            print(summ)
            summ.to_csv(os.path.join(self.resultsFolder,'summ.csv'))
            az.plot_trace(self.trace)
            plt.savefig(os.path.join(self.resultsFolder,'traceplot.jpg'))
            plt.close()

        # df = pm.backends.tracetab.trace_to_dataframe(self.trace)
        az.to_netcdf(data=self.trace,
                     filename=os.path.join(self.resultsFolder, 'trace.nc'))
        # df = self.trace.to_dataframe()
        # df.to_csv(os.path.join(self.resultsFolder, 'trace.csv'))

    def samplePosterior(self):

        self.restoreTrace()

        with self.SNCurveModel:
            NNew = self.gp_ht.conditional("NNew", Xnew=self.SNew)
            lg_σ_f_pred = self.σ_gp.conditional("log_σ_f_pred", Xnew=self.SNew)
            # or to predict the GP plus noise
            y_samples = pm.sample_posterior_predictive(trace=self.trace, var_names=['NNew','log_σ_f_pred'])

        try:
            y_samples = {key: val.tolist() for key, val in y_samples.items()}
        except Exception as e:
            print(e)

        with open(os.path.join(self.resultsFolder,'samples.json'),'w') as file:
            json.dump(y_samples, file)

    def plotGP(self,):

        S = self.S.reshape((-1,1))

        with open(os.path.join(self.resultsFolder, 'samples.json'),'r') as file:
            y_samples = json.load(file)

        y_samples = {key:jnp.array(val) for key, val in y_samples.items()}
        # counts, bins = jnp.histogram(y_samples['damageVals'].flatten())

        # _, ax = plt.subplots(1,1,figsize=(10, 4))
        # ax.hist(bins[:-1], bins, weights=counts)
        # plt.savefig('plots3/damageParam.jpg')
        # plt.close()

        _, axs = plt.subplots(1, 3, figsize=(18, 4))
        # μ_samples = y_samples["NNew"].mean(axis=0)
        μ_samples = y_samples["NNew"]
        σ_samples = np.exp(y_samples["log_σ_f_pred"])

        plot_mean(axs[0], μ_samples, Xnew=self.SNew, ynew=y_samples["NNew"].mean(axis=0), X=S, y=self.log_N)
        plot_var(axs[1], σ_samples ** 2, X=S, Xnew=self.SNew,y_err=1)
        plot_total(axs[2], μ_samples,
                   var_samples=σ_samples ** 2,
                   Xnew=self.SNew, ynew=y_samples['NNew'].mean(axis=0),
                   X_obs=S, y_obs_=self.log_N)

        plt.savefig(os.path.join(self.resultsFolder,'heteroModel.jpg'))
        plt.close()

        _, axs = plt.subplots(1, 2, figsize=(12, 8))

        newMean = vmap(log1exp, in_axes=(0,))(y_samples['NNew'])

        plot_gp_dist(axs[0], newMean, self.SNew[:,None], palette="bone_r")
        axs[0].scatter(self.S, self.log_N, label="Experimental Data")
        axs[0].set_xlabel('Stress/MaxStress')
        axs[0].set_ylabel('log(N)/max(log(N))')
        print(y_samples["log_σ_f_pred"].shape)
        print(self.SNew.shape)
        plot_gp_dist(axs[1], σ_samples, self.SNew[:,None])
        axs[1].set_xlabel('Stress/MaxStress')
        axs[1].set_ylabel('Calculated Variance')
        plt.savefig(os.path.join(self.resultsFolder,'GPDist.jpg'))
        plt.close()

    def restoreTrace(self):
        path = os.path.join(self.resultsFolder, 'trace.nc')

        if os.path.exists(path):
            self.trace = az.from_netcdf(filename=path)
        else:
            self.sampleModel(2000)
        return self.trace
