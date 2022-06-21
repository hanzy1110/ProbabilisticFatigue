import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3.gp.util import plot_gp_dist
from functools import reduce
import jax.numpy as jnp
import jax
import json

from .damageModelGao import gaoModel_debug, gaoModel
from .freqModel import LoadModel
from .models import WohlerCurve

from jax.config import config
config.update("jax_debug_nans", True)


class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jnp.DeviceArray):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self,obj)

def amplitudesToLoads(amplitudes, pf_slope):
    return amplitudes*pf_slope

def log1exp(arr:jnp.ndarray):
    return jnp.log(1+jnp.exp(arr))

def sliceArrs(arr1, arr2, out):
    masks = [jnp.isclose(arr1, val) for val in arr2]
    init = jnp.array([False for _ in masks[0]])
    mask = reduce(lambda x,y: jnp.logical_or(x,y), masks, init)
    # mask = arr1<arr2.max() or arr1>arr2.min()
    return jnp.transpose(out)[mask]

class DamageCalculation:
    def __init__(self, wohlerPath:str, loadPath:str,
                 WohlerObserved:str, loadObserved:str,
                 )->None:

        self.wohlerPath = wohlerPath
        self.loadPath = loadPath
        self.WohlerC = WohlerCurve(resultsFolder=wohlerPath,
                                   observedDataPath=WohlerObserved)

        self.LoadM = LoadModel(resultsFolder=loadPath,
                               observedDataPath=loadObserved,
                               )

        self.WohlerC.build_HeteroskedasticModel()
        self.LoadM.buildMixtureFreqModel()

        self.restoreTraces()

    # @profile
    def restoreTraces(self):
            self.traceWohler = self.WohlerC.restoreTrace()
            self.traceLoads = self.LoadM.restoreTrace()

    def sampleLoads(self, ndraws):
        with self.LoadM.amplitude_model:
            # pf_slope = pm.Normal('pf_slope', mu=5, sigma=0.5) * self.LoadM.maxAmp
            loads = pm.Deterministic('Loads', amplitudesToLoads(self.LoadM.loads, 1))
            self.sample_loads = pm.sample_posterior_predictive(trace=self.traceLoads,
                                                               var_names=['Loads'],
                                                               samples=ndraws)
        loads = self.sample_loads['Loads']
        with open(os.path.join(self.loadPath, 'loadSamples.npz'), 'wb') as file:
            np.savez_compressed(file, loads)

    def transformGPs(self):

        meanSamps, varSamps = list(self.samples.values())
        self.meanSamples = jnp.array(meanSamps)

        self.varianceSamples = jnp.array(varSamps)
        self.totalN = jax.vmap(log1exp, in_axes=(0,))(self.meanSamples)

        SNew = jnp.array(self.WohlerC.SNew.flatten())
        slicedTotal = sliceArrs(SNew, self.amplitudes, self.totalN)

        print('shapes-->')
        print(self.totalN.shape)
        print(slicedTotal.shape)
        self.totalNNotNorm = jnp.transpose(slicedTotal )* self.WohlerC.NMax

        self.Nsamples = jax.vmap(jnp.exp, in_axes=(0,))(self.totalNNotNorm)

    def sampleFatigueLife(self, maxLoads:int):
        loads = list(self.sample_loads.values())[0]
        self.cycles, self.amplitudes = jnp.histogram(loads.flatten(), bins=maxLoads)

        # Transform to amplitudes and then to 0,1 in Wohler Space
        self.amplitudes = self.amplitudes* self.LoadM.maxAmp/self.WohlerC.SMax

        with self.WohlerC.SNCurveModel:
            meanN = self.WohlerC.gp_ht.conditional('meanN', Xnew=self.WohlerC.SNew)
            sigmaN = self.WohlerC.σ_gp.conditional('sigmaN', Xnew=self.WohlerC.SNew)
            # totalN = pm.Deterministic('totalN', meanN + sigmaN)
            self.samples = pm.sample_posterior_predictive(self.traceWohler, var_names=['meanN', 'sigmaN'])

        meanSamples = self.samples['meanN']
        varianceSamples = self.samples['sigmaN']
        with open(os.path.join(self.wohlerPath, 'lifeSamples.npz'), 'wb') as file:
            np.savez_compressed(file, meanSamples, varianceSamples)

        self.transformGPs()

    # @profile
    def restoreFatigueLifeSamples(self, maxLoads):

        if os.path.exists(os.path.join(self.wohlerPath, 'lifeSamples.npz')):

            loads = list(self.sample_loads.values())[0]
            self.cycles, self.amplitudes = jnp.histogram(loads.flatten(), bins=maxLoads)
            # self.amplitudes = self.amplitudes* self.LoadM.maxAmp/self.WohlerC.SMax

            with open(os.path.join(self.wohlerPath, 'lifeSamples.npz'), 'rb') as file:
                samples = np.load(file, allow_pickle=True)
                self.samples = {key:jnp.array(val) for key, val in samples.items()}
            self.transformGPs()

        else:
            self.sampleFatigueLife(maxLoads=maxLoads)

    # @profile
    def restoreLoadSamples(self, ndraws):

        if os.path.exists(os.path.join(self.loadPath, 'loadSamples.npz')):
            with open(os.path.join(self.loadPath, 'loadSamples.npz'), 'rb') as file:
                samples = np.load(file, allow_pickle=True)
                self.sample_loads = {key:jnp.array(val) for key, val in samples.items()}
        else:
            self.sampleLoads(ndraws=ndraws)

    # @profile
    def plotFatigueLifeSamples(self):
        newMean = jax.vmap(log1exp, in_axes=(0,))(self.meanSamples)
        newVar = jax.vmap(jnp.exp, in_axes=(0,))(self.varianceSamples)
        _,ax = plt.subplots(4,1, figsize=(20,14))

        plot_gp_dist(ax[0], self.totalN, self.WohlerC.SNew, palette="bone_r")
        plot_gp_dist(ax[1], newMean, self.WohlerC.SNew, palette="bone_r")
        plot_gp_dist(ax[2], newVar, self.WohlerC.SNew)
        plot_gp_dist(ax[3], self.varianceSamples, self.WohlerC.SNew)

        ax[0].set_ylabel('Cycles to failure')
        ax[0].set_xlabel('Amplitudes')

        ax[1].set_ylabel('Mean')
        ax[2].set_ylabel('Variance')
        ax[3].set_ylabel('Variance (notTransformed)')

        ax[1].set_xlabel('Amplitudes')
        ax[2].set_xlabel('Amplitudes')
        ax[3].set_xlabel('Amplitudes')
        plt.savefig(os.path.join(self.wohlerPath, 'newSamples.jpg'))
        plt.close()
        _,ax = plt.subplots(1,1)
        loads = list(self.sample_loads.values())[0]
        counts, bins = jnp.histogram(loads)
        bins *= self.LoadM.maxAmp/self.WohlerC.NMax
        ax.hist(bins[:-1], bins, weights=counts)
        ax.set_xlabel('Loads [Sampled]')
        ax.set_ylabel('Density')

        plt.savefig(os.path.join(self.wohlerPath, 'loadSamples.jpg'))
        plt.close()

    def calculateDamage(self):

        cycles = jnp.array(self.cycles)
        Nf = jnp.array(self.Nsamples)

        damageFun = jax.vmap(gaoModel, in_axes=(None,0))
        self.damages = damageFun(cycles, Nf)

        _, ax = plt.subplots(1,1, figsize=(12,8))
        counts, bins = jnp.histogram(self.damages)
        ax.hist(bins[:-1], bins, weights=counts)
        plt.savefig(os.path.join(self.wohlerPath, 'damageHist.jpg'))
        plt.close()

        return self.damages

    def calculateDamage_debug(self):

        cycles = jnp.array(self.cycles)
        Nf = jnp.array(self.Nsamples)

        self.damages = np.zeros_like(self.Nsamples.tolist())
        new_arr = np.array(self.Nsamples.tolist())
        lnNew = np.array(self.totalN.tolist())

        for i, (N, lnN) in enumerate(zip(new_arr, lnNew)):
            self.damages[i] = gaoModel_debug(cycles, N, lnN)

        # _, ax = plt.subplots(1,1, figsize=(12,8))
        # counts, bins = jnp.histogram(self.damages)
        # ax.hist(bins[:-1], bins, weights=counts)
        # plt.savefig(os.path.join(self.wohlerPath, 'damageHist.jpg'))
        # plt.close()

        return self.damages

    def sample_model(self, model:str, ndraws):
        if model == 'wohler':
            self.WohlerC.sampleModel(ndraws)
            self.WohlerC.samplePosterior()
            self.WohlerC.plotGP()

        elif model == 'loads':
            self.LoadM.sampleModel(ndraws)

