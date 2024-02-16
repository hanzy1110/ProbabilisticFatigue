# %%
import os
import pathlib
import numpy as np
import scipy
import pymc as pm
import nutpie
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from pymc.gp.util import plot_gp_dist
import json

from .plotUtils import plot_mean, plot_var, plot_total, get_ℓ_prior
import jax.numpy as jnp
from jax import vmap, jit

import scienceplots

plt.style.use(["science", "ieee"])


def logistic(x, a, x0, c):
    # a is the slope, x0 is the location
    return c * (1 - pm.math.invlogit(a * (x - x0)))


def log1exp(arr: jnp.ndarray):
    return jnp.log(1 + jnp.exp(arr))


class WohlerCurve:
    def __init__(
        self, resultsFolder: pathlib.Path, observedDataPath: pathlib.Path
    ) -> None:
        if not os.path.exists(resultsFolder):
            os.mkdir(resultsFolder)

        self.results_folder = resultsFolder
        # Both are given as rows!
        SN_data = scipy.io.loadmat(observedDataPath)
        self.log_N = SN_data["X"].flatten()
        self.S = SN_data["Y"].flatten()

    def NormalizeData(self, plotExp=True):
        log_n_prev = self.log_N
        s = self.S
        self.NMax = np.log(self.log_N.max() * 1e6)
        self.log_N = np.log(self.log_N * 1e6) / self.NMax

        self.SMax = self.S.max()
        self.S /= self.SMax
        self.SNew = np.linspace(self.S.min(), self.S.max(), 100)[:, None]
        # self.SNew = self.S.reshape(-1, 1)

        if plotExp:
            _, (ax, ax1) = plt.subplots(1, 2)
            ax.scatter(self.S, self.log_N)
            ax.set_ylabel(r"$log(N)/log(N_{Max})$")
            ax.set_xlabel(r"$\sigma/\sigma_{max}$")
            ax1.scatter(s, log_n_prev)
            ax1.set_ylabel(r"$N$")
            ax1.set_xlabel(r"$\sigma$")
            plt.savefig(os.path.join(self.results_folder, "experimental.jpg"))
            plt.close()

    def build_HeteroskedasticModel(self):
        self.NormalizeData(plotExp=True)
        S = self.S.reshape((-1, 1))

        l_mu, l_sigma = [stat for stat in get_ℓ_prior(self.S)]

        with pm.Model() as self.SNCurveModel:
            l_scale = pm.Gamma(r"$l_{scale}$", mu=l_mu, sigma=l_sigma)
            eta = pm.Gamma(r"$\eta$", alpha=2, beta=1)

            cov = eta**2 * pm.gp.cov.Exponential(
                input_dim=1, ls=l_scale
            ) + pm.gp.cov.WhiteNoise(sigma=1e-6)

            self.gp_ht = pm.gp.Latent(cov_func=cov)
            mu_f = self.gp_ht.prior(r"$\mu_f$", X=S)
            mu_f_tr = pm.Deterministic(r"$\mu_{ftr}$", pm.math.log1pexp(mu_f))
            σ_ℓ = pm.Gamma(r"$\sigma_l$", mu=l_mu, sigma=l_sigma)
            σ_η = pm.Gamma(r"$\sigma_{\eta}$", alpha=2, beta=1)

            σ_cov = σ_η**2 * pm.gp.cov.ExpQuad(
                input_dim=1, ls=σ_ℓ
            ) + pm.gp.cov.WhiteNoise(sigma=1e-6)

            self.σ_gp = pm.gp.Latent(cov_func=σ_cov)
            σ_f = self.σ_gp.prior(r"$lg_{\sigma f}$", X=S)
            σ_f = pm.Deterministic(r"$\sigma_f$", pm.math.exp(σ_f))

            nu = pm.Gamma(r"$\nu$", alpha=2, beta=1)
            lik_ht = pm.StudentT(
                "lik_ht", nu=nu, mu=mu_f_tr, sigma=σ_f, observed=self.log_N
            )

    def sampleModel(self, ndraws):
        print("Sampling WOHLER MODEL")
        # with self.SNCurveModel:
        #     self.trace = pm.sample_smc(draws=ndraws, cores = 4, return_inferencedata=True)
        # self.trace = pm.sample(draws=ndraws, chains=4, tune=2000, target_accept=0.97)

        compiled_model = nutpie.compile_pymc_model(self.SNCurveModel)
        self.trace = nutpie.sample(compiled_model, draws=ndraws, tune=2000, chains=4)
        az.to_netcdf(
            data=self.trace, filename=self.results_folder / "SN_MODEL_TRACE.nc"
        )
        # if not isinstance(self.trace, az.InferenceData):
        #     self.trace = az.convert_to_inference_data(self.trace)
        # self.trace = pm.sample()
        summ = az.summary(self.trace)
        print(summ)
        summ.to_csv(os.path.join(self.results_folder, "SN_MODEL_SUMM.csv"))
        az.plot_trace(self.trace)
        plt.savefig(os.path.join(self.results_folder, "SN_MODEL_TRACEPLOT.jpg"))
        plt.close()

    def samplePosterior(self):
        self.restoreTrace()

        with self.SNCurveModel:
            NNew = self.gp_ht.conditional("NNew", Xnew=self.SNew)
            lg_σ_f_pred = self.σ_gp.conditional("log_σ_f_pred", Xnew=self.SNew)
            # or to predict the GP plus noise
            y_samples = pm.sample_posterior_predictive(
                trace=self.trace, var_names=["NNew", "log_σ_f_pred"]
            )

        # try:
        #     y_samples = {key: val.tolist() for key, val in y_samples.items()}
        # except Exception as e:
        #     print(e)

        az.to_netcdf(data=y_samples, filename=self.results_folder / "SN_SAMPLES.nc")
        # with open(self.results_folder / "SN_SAMPLES.json", "w") as file:
        #     json.dump(y_samples, file)

    def plotGP(
        self,
    ):
        S = self.S.reshape((-1, 1))

        with open(
            os.path.join(self.results_folder, "lifeSamples_20.npz"), "rb"
        ) as file:
            samples = np.load(file, allow_pickle=True)
            y_samples = {key: jnp.array(val) for key, val in samples.items()}

        # with open(self.results_folder / "SN_SAMPLES.json", "r") as file:
        #     y_samples = json.load(file)

        # y_samples = {key: jnp.array(val) for key, val in y_samples.items()}
        # counts, bins = jnp.histogram(y_samples['damageVals'].flatten())

        # _, ax = plt.subplots(1,1,figsize=(10, 4))
        # ax.hist(bins[:-1], bins, weights=counts)
        # plt.savefig('plots3/damageParam.jpg')
        # plt.close()

        _, axs = plt.subplots(1, 3, figsize=(18, 4))
        # μ_samples = y_samples["NNew"].mean(axis=0)
        μ_samples = y_samples["NNew"]
        σ_samples = np.exp(y_samples["log_σ_f_pred"])

        plot_mean(
            axs[0],
            μ_samples,
            Xnew=self.SNew,
            ynew=y_samples["NNew"].mean(axis=0),
            X=S,
            y=self.log_N,
        )
        plot_var(axs[1], σ_samples**2, X=S, Xnew=self.SNew, y_err=1)
        plot_total(
            axs[2],
            μ_samples,
            var_samples=σ_samples**2,
            Xnew=self.SNew,
            ynew=y_samples["NNew"].mean(axis=0),
            X_obs=S,
            y_obs_=self.log_N,
        )

        plt.savefig(self.results_folder / "heteroModel.jpg", dpi=600)
        plt.close()

        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(3.3, 3.3)

        newMean = vmap(log1exp, in_axes=(0,))(y_samples["NNew"])

        plot_gp_dist(axs[0], newMean, self.SNew[:, None], palette="bone_r")
        axs[0].scatter(self.S, self.log_N, label="Datos Experimentales")
        axs[0].set_xlabel(r"$\sigma/\sigma_{max}$")
        axs[0].set_ylabel(r"log(N)/max(log(N))")
        print(y_samples["log_σ_f_pred"].shape)
        print(self.SNew.shape)
        plot_gp_dist(axs[1], σ_samples, self.SNew[:, None])
        axs[1].set_xlabel(r"$\sigma/\sigma_{max}$")
        axs[1].set_ylabel("Variancia Calculada")
        plt.savefig(self.results_folder / "WOHLER_GPDIST.jpg")
        plt.close()

    def restoreTrace(self, ndraws):
        path = self.results_folder / "SN_MODEL_TRACE.nc"

        if os.path.exists(path):
            self.trace = az.from_netcdf(filename=path)
        else:
            self.sampleModel(ndraws)
        return self.trace
