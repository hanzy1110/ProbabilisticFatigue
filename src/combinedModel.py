import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
from pymc.gp.util import plot_gp_dist
from functools import reduce, partial
import jax.numpy as jnp
from jax.scipy import stats
import jax
from jax import Array
from tqdm import tqdm

from time import perf_counter
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from typing import Dict, Any, List

from .plotUtils import plot_mean, plot_total, plot_var
from .damageModelGao import gaoModel_debug, gaoModel, minerRule
from .aeran_model import aeran_model
from .freqModel import LoadModel
from .models import WohlerCurve
from .stressModel import CableProps, PFSlopeModel

from jax.config import config
import matplotlib as mpl

import scienceplots

plt.style.use(["science", "ieee"])

mpl.rcParams["agg.path.chunksize"] = 10000
# plt.style.use(['science', 'ieee'])
# config.update("jax_debug_nans", True)


def get_batch(arr, size, batch_dim=0, max_batches=10):
    n_batches = arr.shape[batch_dim] // size

    for i in range(n_batches):
        if i > max_batches:
            break
        yield arr.take(indices=jnp.arange(i * size, (i + 1) * size), axis=batch_dim)


def get_random_params(x, y, idx, size):
    y_slice = y[idx, :]
    x_slice = x[idx, :]
    return np.random.choice(x_slice, size=size), np.random.choice(y_slice, size=size)


def amplitudesToLoads(amplitudes, pf_slope):
    return amplitudes * pf_slope


def log1exp(arr: jnp.ndarray):
    return jnp.log(1 + jnp.exp(arr))
    # return jnp.log(jnp.exp(arr) - 1)


def log1expM(arr: jnp.ndarray):
    return jnp.log(jnp.exp(arr) - 1)


@jax.jit
def fillArray(arr: Array, num: int = 5):
    return reduce(lambda x, y: jnp.hstack((x, jnp.linspace(x.max(), y, num=num))), arr)


def completeArray(arr: Array, num):
    if arr.max() < 1:
        return jnp.hstack((arr, jnp.linspace(arr.max(), 1, num)))
    else:
        return arr


def sliceArrs(dct, out):
    # masks = [jnp.isclose(dct["arr1"], val) for val in dct["arr2"]]
    # init = jnp.array([False for _ in masks[0]])
    # mask = reduce(lambda x, y: jnp.logical_or(x, y), masks, init)

    mask = jnp.where(dct["arr1"] < dct["arr2"].max())

    # return jnp.transpose(out)[mask]
    return out[mask][:-1, ...]


def delete_live_arrays():
    print("Deleting data from devices...")
    for arr in jax.live_arrays():
        arr.delete()
    # for device in devices():
    #     lbs = device.live_buffers()
    #     for lb in lbs:
    #         lb.delete()


class DamageCalculation:
    def __init__(
        self,
        wohlerPath: pathlib.Path,
        loadPath: pathlib.Path,
        WohlerObserved: pathlib.Path,
        loadObserved: pathlib.Path,
        PFObserved: pathlib.Path,
        cableProps: Dict[str, Any],
        Tpercentage: int,
    ) -> None:
        # Check units to get the correct value
        self.Tpercentage = Tpercentage
        self.CableProps = CableProps(**cableProps)
        self.CableProps.getLambda(cableProps["T"])
        self.meanPFSlope = self.CableProps.PFSlope()

        print("PFSlope", self.meanPFSlope)
        self.PFSlopeModel = PFSlopeModel(
            PFObserved, cableProps, Tpercentage, wohlerPath
        )
        self.wohler_path = wohlerPath
        self.loadPath = loadPath
        self.WohlerC = WohlerCurve(
            resultsFolder=wohlerPath, observedDataPath=WohlerObserved
        )

        self.LoadM = LoadModel(
            resultsFolder=loadPath,
            observedDataPath=loadObserved,
            Tpercentage=Tpercentage,
            ydata=PFObserved,
            cableProps=cableProps,
        )

        self.WohlerC.build_HeteroskedasticModel()
        self.LoadM.buildMixtureFreqModel()

        # for load sampling purposes
        self.varCoeff = 0.1

    # #@profile
    def restoreTraces(self, ndraws_wohler):
        self.traceWohler = self.WohlerC.restoreTrace(ndraws=ndraws_wohler)
        self.traceLoads = self.LoadM.restoreTrace()
        self.tracePF = self.PFSlopeModel.restoreTrace()

    def restoreFatigueLifeSamples(self, maxLoads):
        self.joinAmplitudesStress(maxLoads)

        if os.path.exists(
            os.path.join(self.wohler_path, f"lifeSamples_{self.Tpercentage}.npz")
        ):
            self.loadLifeSamples()
        else:
            self.sampleFatigueLife(maxLoads=maxLoads)

        self.transformGPs()

    # @profile
    def restoreLoadSamples(self, ndraws):
        if os.path.exists(
            os.path.join(self.loadPath, f"loadSamples_{self.Tpercentage}.npz")
        ):
            self.loadLoadSamples()
        else:
            self.sampleLoads(ndraws=ndraws)

    # @profile
    def sampleLoads(self, ndraws):
        self.LoadM.buildMixtureFreqModel()
        print("Sampling Loads")
        maxAmp = self.LoadM.maxAmp * 1e-3  # mum to mm
        with self.LoadM.amplitude_model:
            # pf_slope = pm.Normal('pf_slope', mu=self.meanPFSlope,
            #                      sigma=self.meanPFSlope*self.varCoeff)*maxAmp
            # pf_slope = 1
            # pfSlope * maxAmp transforms it from normalized to sigma
            loads = pm.Deterministic(
                "Loads",
                amplitudesToLoads(self.LoadM.loads, self.LoadM.pfSlope * maxAmp),
            )

            self.sample_loads = pm.sample_posterior_predictive(
                trace=self.traceLoads, var_names=["Loads"]
            )

        loads = self.sample_loads.posterior_predictive["Loads"]
        print("Load Shapes-->", loads.shape)
        with open(self.loadPath / f"loadSamples_{self.Tpercentage}.npz", "wb") as file:
            np.savez_compressed(file, loads)

    def sampleFatigueLife(self, maxLoads: int):
        print(f"Sampling Fatigue Life, maxLoads {maxLoads}")
        Xnew = np.array(self.SNew.reshape((-1, 1)))
        with self.WohlerC.SNCurveModel:
            meanN = self.WohlerC.gp_ht.conditional("meanN", Xnew=Xnew)
            sigmaN = self.WohlerC.σ_gp.conditional("sigmaN", Xnew=Xnew)
            # totalN = pm.Deterministic('totalN', meanN + sigmaN)
            self.samples = pm.sample_posterior_predictive(
                self.traceWohler, var_names=["meanN", "sigmaN"]
            )

        meanSamples = self.samples.posterior_predictive["meanN"]
        varianceSamples = self.samples.posterior_predictive["sigmaN"]
        print("shapes-->")
        print(meanSamples.shape)
        print(varianceSamples.shape)

        with open(
            os.path.join(self.wohler_path, f"lifeSamples_{self.Tpercentage}.npz"), "wb"
        ) as file:
            np.savez_compressed(file, meanSamples, varianceSamples)

    # @profile
    def calculate_damage(self, cycles_per_year, year, plot: bool = False):
        print("=/" * 30)
        print("Damage According to Aeran")
        # cycles = jnp.array(self.cycles, dtype=jnp.float16)[:10, :]
        # CLEARLY NEEDS WORK!!!
        cycles = jnp.array(self.cycles, dtype=jnp.float32) * cycles_per_year

        n_cycles = cycles.sum(axis=1)
        print(f"Total Cycles: {n_cycles.mean()}")
        Nf = jnp.array(self.Nsamples, dtype=jnp.float32)
        lnNf = jnp.array(self.slicedTotal.T, dtype=jnp.float32)

        if Nf.shape[1] < self.amplitudes.shape[1]:
            sigma_i = jnp.array(self.amplitudes.T[:, : Nf.shape[1]], dtype=jnp.float32)
        else:
            sigma_i = jnp.array(self.amplitudes.T, dtype=jnp.float32)
            Nf = Nf[:, : sigma_i.shape[1]]

        print(f"cycles shape: {cycles.shape} ")
        print(f"Nf shape: {Nf.shape} ")
        print(f"lnNF shape: {lnNf.shape} ")
        print(f"sigma_i shape: {sigma_i.shape} ")

        damageFun = jax.vmap(aeran_model, in_axes=(None, 1, 1))
        coolDamageFun = jax.vmap(lambda x: damageFun(x, Nf, sigma_i), in_axes=(0,))

        tot_damages = None

        BATCH_SIZE = 50
        MAX_BATCHES = 1500
        i = 0
        for i, batch in tqdm(
            enumerate(
                get_batch(cycles, BATCH_SIZE, batch_dim=0, max_batches=MAX_BATCHES)
            ),
            total=cycles.shape[0] // BATCH_SIZE,
            leave=True,
        ):
            tot_damages = np.array(coolDamageFun(batch).flatten())
            # print(f"res shape : {tot_damages.shape}")

            with open(
                self.loadPath / f"tot_damages_year{year}_batch_{i}.npz", "wb"
            ) as file:
                np.savez_compressed(file, tot_damages)
        # delete_live_arrays()
        return i

    def calculate_damage_miner(self, cycles_per_year, _iter, plot=False):
        print("=/" * 30)
        print("Damage According to Miner")

        # cycles = jnp.array(self.cycles)

        cycles = jnp.array(self.cycles, dtype=jnp.float16) * cycles_per_year
        Nf = jnp.array(self.Nsamples, dtype=jnp.float16)[:-1, :]
        # Nf = jnp.array(self.Nsamples)[..., :-2]

        n_cycles = cycles.sum(axis=1)
        print(f"Total Cycles: {n_cycles.mean()}")

        damageFun = jax.vmap(minerRule, in_axes=(None, 1))
        coolDamageFun = jax.vmap(lambda x: damageFun(x, Nf), in_axes=(0,))

        init = perf_counter()
        self.damages = coolDamageFun(cycles).flatten()
        end = perf_counter()

        print(f"Time passed: {end-init}")

        nanFrac = len(self.damages[jnp.isnan(self.damages)]) / len(self.damages)
        print(f"NaN Damage Fraction: {nanFrac}")

        if nanFrac < 0.5:
            if plot:
                fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                ax[0].plot(self.damages)
                ax[0].set_title("Damage according to Miners rule")
                try:
                    counts, bins = jnp.histogram(
                        self.damages[jnp.where(~jnp.isnan(self.damages))],
                        bins=50,
                        density=True,
                    )
                    ax[1].hist(bins[:-1], bins, weights=counts)
                except Exception as e:
                    print(e)

                plt.savefig(self.wohler_path / "damageHistMiner.jpg", dpi=600)
                plt.close(fig)
            return self.damages

        return None

    def calculate_damage_debug(
        self,
        cycles_per_year,
    ):
        # cycles = jnp.array(self.cycles)
        print("=/" * 30)
        print("Damage According to Aeran")
        # cycles = jnp.array(self.cycles, dtype=jnp.float16)[:10, :]
        cycles = jnp.array(self.cycles, dtype=jnp.float32) * cycles_per_year
        n_cycles = cycles.sum(axis=1)
        print(f"Total Cycles: {n_cycles.mean()}")
        Nf = jnp.array(self.Nsamples, dtype=jnp.float32)
        lnNf = jnp.array(self.slicedTotal.T, dtype=jnp.float32)
        sigma_i = jnp.array(self.amplitudes.T[:, : Nf.shape[1]], dtype=jnp.float32)
        shape = (Nf.shape[0],)
        print(f"cycles shape: {cycles.shape} ")
        print(f"Nf shape: {Nf.shape} ")
        print(f"lnNF shape: {lnNf.shape} ")
        print(f"sigma_i shape: {sigma_i.shape} ")

        dmgs = []
        for i, c in enumerate(cycles):
            Ni, sigma_i = get_random_params(Nf, sigma_i, idx=i, size=10)
            dmgs.append(aeran_model(c, Ni, sigma_i, shape=c.shape))

        return np.hstack(dmgs)

    def loadLifeSamples(self):
        with open(
            os.path.join(self.wohler_path, f"lifeSamples_{self.Tpercentage}.npz"), "rb"
        ) as file:
            samples = np.load(file, allow_pickle=True)
            self.samples = {key: jnp.array(val) for key, val in samples.items()}

    def loadLoadSamples(self):
        with open(
            os.path.join(self.loadPath, f"loadSamples_{self.Tpercentage}.npz"), "rb"
        ) as file:
            samples = np.load(file, allow_pickle=True)
            self.sample_loads = {key: jnp.array(val) for key, val in samples.items()}
            shapes = [val.shape for val in self.sample_loads.values()]
            print(f"SAMPLE LOADS SHAPE =>> {shapes}")

    # @profile
    def transformGPs(self):
        print("Transforming GPs")
        meanSamps, varSamps = list(self.samples.values())
        # mean over the chains
        self.meanSamples = jnp.array(meanSamps).mean(axis=0)

        self.varianceSamples = jnp.array(varSamps).mean(axis=0)
        self.totalN = jax.vmap(log1exp, in_axes=(1,))(self.meanSamples)

        print("meanSamples shape-->", self.meanSamples.shape)
        print("Total N shape-->", self.totalN.shape)

        # sliceArrsvmap = jax.jit(
        #     jax.vmap(
        #         lambda dct: sliceArrs(dct, self.totalN),
        #         in_axes=({"arr1": 0, "arr2": 0},),
        #     )
        # )

        # transposeT = jnp.transpose(self.totalN)
        dct = {"arr1": self.SNew, "arr2": self.amplitudes.mean(axis=0)}
        # self.slicedTotal = sliceArrsvmap(dct)

        self.slicedTotal = sliceArrs(dct, self.totalN)

        self.slicedTotal = jnp.transpose(self.slicedTotal)
        self.totalNNotNorm = self.slicedTotal * self.WohlerC.NMax

        self.Nsamples = jax.vmap(jnp.exp, in_axes=(1,))(self.totalNNotNorm)

        print("slicedTotal shape-->", self.slicedTotal.shape)
        print("Nsamples shape -->", self.Nsamples.shape)

    # @profile
    def joinAmplitudesStress(self, maxLoads):
        print("Join amplitude and stress...")
        loads = list(self.sample_loads.values())[0].mean(axis=0)
        # loads = self.sample_loads.posterior_predictive.mean("chain")["Loads"]
        vHisto = jax.vmap(
            lambda x: jnp.histogram(x, bins=maxLoads, density=True), in_axes=(1,)
        )
        self.cycles, self.amplitudes = vHisto(loads)

        fig, ax = plt.subplots(1, 1)
        for i in range(5):
            ax.stairs(self.cycles[i, :], self.amplitudes[i, :])
        ax.set_xlabel("Amplitudes")
        ax.set_ylabel("Sampled density")

        plt.savefig(self.wohler_path / "sampled_loads.png", dpi=600)

        # Transform to amplitudes and then to 0,1 in Wohler Space
        self.amplitudes /= self.WohlerC.SMax

        self.SNew = completeArray(self.amplitudes[0, :], num=30)
        # Just take the mean samples...
        # self.SNew = self.amplitudes.mean(axis=0)
        print(f"SNew shape: {self.SNew.shape}")

    def sample_model(self, model: str, ndraws):
        if model == "wohler":
            self.WohlerC.sampleModel(ndraws)
            self.WohlerC.samplePosterior()
            self.WohlerC.plotGP()

        elif model == "loads":
            self.LoadM.sampleModel(ndraws)

    def plotLoadSamples(self):
        print("Plotting Samples...")
        loads = list(self.sample_loads.values())[0]
        loads = jnp.array(loads)
        vHisto = jax.vmap(lambda x: jnp.histogram(x), in_axes=(1,))
        counts, bins = vHisto(loads)
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        for i, (count, bin_) in enumerate(zip(counts, bins)):
            ax.hist(bin_[:-1], bin_, weights=count, density=True)
            if i > 200:
                break
        plt.savefig(os.path.join(self.wohler_path, "loadSample2.jpg"))
        plt.close()

    def plotFatigueLifeSamples(self):
        newMean = jax.vmap(log1exp, in_axes=(0,))(self.meanSamples)
        newVar = jax.vmap(jnp.exp, in_axes=(0,))(self.varianceSamples)
        _, ax = plt.subplots(4, 1, figsize=(20, 14))

        plot_gp_dist(ax[0], self.totalN.T, self.SNew, palette="bone_r")
        plot_gp_dist(ax[1], newMean, self.SNew, palette="bone_r")
        plot_gp_dist(ax[2], newVar, self.SNew)
        plot_gp_dist(ax[3], self.varianceSamples, self.SNew)

        ax[0].set_ylabel("Cycles to failure")
        ax[0].set_xlabel("Amplitudes")

        ax[1].set_ylabel("Mean")
        ax[2].set_ylabel("Variance")
        ax[3].set_ylabel("Variance (notTransformed)")

        ax[1].set_xlabel("Amplitudes")
        ax[2].set_xlabel("Amplitudes")
        ax[3].set_xlabel("Amplitudes")
        plt.savefig(os.path.join(self.wohler_path, "newSamples.jpg"))
        plt.close()
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(3.3, 3.3)
        loads = list(self.sample_loads.values())[0]
        counts, bins = jnp.histogram(loads)
        bins *= self.LoadM.maxAmp / self.WohlerC.NMax
        ax.hist(bins[:-1], bins, weights=counts)
        ax.set_xlabel("Loads [Sampled]")
        ax.set_ylabel("Density")

        plt.savefig(self.wohler_path / "loadSamples.jpg", dpi=600)
        plt.close()

        fig, axs = plt.subplots(3, 1)
        fig.set_size_inches(6.3, 6.3)
        plot_mean(
            axs[0],
            newMean,
            Xnew=self.SNew,
            ynew=newMean.mean(axis=0),
            X=self.WohlerC.S,
            y=self.WohlerC.log_N,
        )
        plot_var(axs[1], newVar, X=self.WohlerC.S, Xnew=self.SNew, y_err=1)
        plot_total(
            axs[2],
            newMean,
            var_samples=newVar,
            Xnew=self.SNew,
            ynew=newMean.mean(axis=0),
            X_obs=self.WohlerC.S,
            y_obs_=self.WohlerC.log_N,
        )

        plt.savefig(self.wohler_path / "heteroModel.jpg", dpi=600)
        plt.close()
