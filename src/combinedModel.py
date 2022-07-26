import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
from pymc3.gp.util import plot_gp_dist
from functools import reduce, partial
import jax.numpy as jnp
from jax.scipy import stats
import jax
import json

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from typing import Dict, Any, List
from .damageModelGao import gaoModel_debug, gaoModel, minerRule
from .freqModel import LoadModel
from .models import WohlerCurve
from .stressModel import CableProps

from jax.config import config
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
plt.style.use(['science', 'ieee'])
# config.update("jax_debug_nans", True)

def amplitudesToLoads(amplitudes, pf_slope):
    return amplitudes*pf_slope

def log1exp(arr:jnp.ndarray):
    return jnp.log(1+jnp.exp(arr))

def log1expM(arr:jnp.ndarray):
    return jnp.log(jnp.exp(arr)-1)

@jax.jit
def fillArray(arr:jnp.DeviceArray, num:int=5):
    return reduce(lambda x,y: jnp.hstack((x,
                             jnp.linspace(x.max(), y, num=num))),
                             arr)

def completeArray(arr:jnp.DeviceArray,num):
    if arr.max()<1:
        return jnp.hstack((arr, jnp.linspace(arr.max(), 1, num)))
    else:
        return arr

def sliceArrs(dct, out):
    masks = [jnp.isclose(dct['arr1'], val) for val in dct['arr2']]
    init = jnp.array([False for _ in masks[0]])
    mask = reduce(lambda x,y: jnp.logical_or(x,y), masks, init)
    # return jnp.transpose(out)[jnp.where(mask, size=out.shape[1],fill_value=-1)]
    return jnp.transpose(out)[mask]
    # return jnp.where(mask.flatten(), jnp.transpose(out), -10)

class DamageCalculation:
    def __init__(self, wohlerPath:str, loadPath:str,
                 WohlerObserved:str, loadObserved:str,
                 cableProps:Dict[str,Any])->None:
        #Check units to get the correct value

        self.CableProps = CableProps(**cableProps)
        self.CableProps.getLambda(cableProps['T'])
        self.meanPFSlope = self.CableProps.PFSlope()

        print("PFSlope", self.meanPFSlope)

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

        # for load sampling purposes
        self.varCoeff = 0.1

    # #@profile
    def restoreTraces(self):
        self.traceWohler = self.WohlerC.restoreTrace()
        self.traceLoads = self.LoadM.restoreTrace()

    def restoreFatigueLifeSamples(self, maxLoads):

        self.joinAmplitudesStress(maxLoads)

        if os.path.exists(os.path.join(self.wohlerPath, 'lifeSamples.npz')):
            self.loadLifeSamples()
        else:
            self.sampleFatigueLife(maxLoads=maxLoads)

        self.transformGPs()

    #@profile
    def restoreLoadSamples(self, ndraws):
        if os.path.exists(os.path.join(self.loadPath, 'loadSamples.npz')):
            self.loadLoadSamples()
        else:
            self.sampleLoads(ndraws=ndraws)

    #@profile
    def sampleLoads(self, ndraws):
        print("Sampling Loads")
        maxAmp = self.LoadM.maxAmp * 1e-3 #mum to mm
        with self.LoadM.amplitude_model:
            pf_slope = pm.Normal('pf_slope', mu=self.meanPFSlope,
                                 sigma=self.meanPFSlope*self.varCoeff)*maxAmp
            # pf_slope = 1
            loads = pm.Deterministic('Loads', amplitudesToLoads(self.LoadM.loads,pf_slope))
            self.sample_loads = pm.sample_posterior_predictive(trace=self.traceLoads,
                                                               var_names=['Loads'],
                                                               samples=ndraws)
        loads = self.sample_loads['Loads']
        print('Load Shapes-->', loads.shape)
        with open(os.path.join(self.loadPath, 'loadSamples.npz'), 'wb') as file:
            np.savez_compressed(file, loads)

        #@profile
    def sampleFatigueLife(self, maxLoads:int):

        print("Sampling Fatigue Life")
        Xnew = self.SNew.reshape((-1,1))
        with self.WohlerC.SNCurveModel:
            meanN = self.WohlerC.gp_ht.conditional('meanN', Xnew=Xnew)
            sigmaN = self.WohlerC.σ_gp.conditional('sigmaN', Xnew=Xnew)
            # totalN = pm.Deterministic('totalN', meanN + sigmaN)
            self.samples = pm.sample_posterior_predictive(self.traceWohler, var_names=['meanN','sigmaN'])

        meanSamples = self.samples['meanN']
        varianceSamples = self.samples['sigmaN']
        print('shapes-->')
        print(meanSamples.shape)
        print(varianceSamples.shape)

        with open(os.path.join(self.wohlerPath, 'lifeSamples.npz'), 'wb') as file:
            np.savez_compressed(file, meanSamples, varianceSamples)

    # @profile
    def calculateDamage(self, scaleFactor, _iter):

        cycles = jnp.array(self.cycles)
        n_cycles = cycles.sum(axis=1)
        print(f'Total Cycles: {n_cycles.mean() * scaleFactor}')
        cycles *= scaleFactor

        Nf = jnp.array(self.Nsamples)
        lnNf = jnp.array(self.slicedTotal)

        damageFun = jax.vmap(gaoModel, in_axes=(None,0,0))
        coolDamageFun = jax.vmap(lambda x: damageFun(x, Nf, lnNf), in_axes=(0,))
        self.damages = coolDamageFun(cycles).flatten()

        _, ax = plt.subplots(1,1, figsize=(12,8))
        # ax[0].plot(self.damages)

        ax.set_title(f'Damage according to Gao Model at N: {cycles.sum(axis=1)[0]}')
        nanFrac = len(self.damages[jnp.isnan(self.damages)])/len(self.damages)
        print(f'NaN Damage Fraction: {nanFrac}')
        if np.isclose(nanFrac,1, rtol=1e-1):
            pass
        else:
            counts, bins = jnp.histogram(self.damages[jnp.where(~jnp.isnan(self.damages))],
                                         density=True, bins=20)
            N, bins, conts = ax.hist(bins[:-1], bins, weights=counts)
            # kde = stats.gaussian_kde(self.damages[~jnp.isnan(self.damages)])
            # dSamps = kde.evaluate(bins)
            # ax.plot(bins, dSamps)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=N.max()))
            ax.set_ylabel('Percentage Damage')
            ax.set_xlabel('Damage')
            # ax.hist(self.damages[~jnp.isnan(self.damages)], density=True, bins=20)

            plt.savefig(os.path.join(self.wohlerPath, f'damageHist_{_iter}.jpg'))
            plt.close()

            return self.damages
        return None

    def calculateDamageMiner(self):

        cycles = jnp.array(self.cycles)
        Nf = jnp.array(self.Nsamples)
        lnNf = jnp.array(self.slicedTotal)

        damageFun = jax.vmap(minerRule, in_axes=(None,0))
        coolDamageFun = jax.vmap(lambda x: damageFun(x, Nf), in_axes=(0,))
        self.damages = coolDamageFun(cycles).flatten()

        nanFrac = len(self.damages[jnp.isnan(self.damages)])/len(self.damages)
        print(f'NaN Damage Fraction: {nanFrac}')

        _, ax = plt.subplots(2,1, figsize=(12,8))
        ax[0].plot(self.damages)
        ax[0].set_title('Damage according to Miners rule')
        try:
            counts, bins = jnp.histogram(self.damages[jnp.where(~jnp.isnan(self.damages))],
                                         bins=50, density=True)
            ax[1].hist(bins[:-1], bins, weights=counts)
        except Exception as e:
            print(e)

        plt.savefig(os.path.join(self.wohlerPath, 'damageHist.jpg'))
        plt.close()

        return self.damages


    def calculateDamage_debug(self):

        # cycles = jnp.array(self.cycles)
        cycles = np.array(self.cycles.tolist())
        # Nf = jnp.array(self.Nsamples)

        self.damages = np.zeros_like(self.Nsamples.tolist())
        new_arr = np.array(self.Nsamples.tolist())
        lnNew = np.array(self.slicedTotal.tolist())

        for i, (N, lnN) in enumerate(zip(new_arr, lnNew)):
            print('New!')
            self.damages[i] = gaoModel_debug(cycles, N, lnN)

        # _, ax = plt.subplots(1,1, figsize=(12,8))
        # counts, bins = jnp.histogram(self.damages)
        # ax.hist(bins[:-1], bins, weights=counts)
        # plt.savefig(os.path.join(self.wohlerPath, 'damageHist.jpg'))
        # plt.close()

        return self.damages

    def loadLifeSamples(self):
        with open(os.path.join(self.wohlerPath, 'lifeSamples.npz'), 'rb') as file:
            samples = np.load(file, allow_pickle=True)
            self.samples = {key:jnp.array(val) for key, val in samples.items()}

    def loadLoadSamples(self):
        with open(os.path.join(self.loadPath, 'loadSamples.npz'), 'rb') as file:
            samples = np.load(file, allow_pickle=True)
            self.sample_loads = {key:jnp.array(val) for key, val in samples.items()}

    #@profile
    def transformGPs(self):
        print("Transforming GPs")
        meanSamps, varSamps = list(self.samples.values())
        self.meanSamples = jnp.array(meanSamps)

        self.varianceSamples = jnp.array(varSamps)
        self.totalN = jax.vmap(log1exp, in_axes=(0,))(self.meanSamples)

        print("meanSamples shape-->",self.meanSamples.shape)
        print("Total N shape-->",self.totalN.shape)

        sliceArrsvmap = jax.jit(jax.vmap(lambda dct: sliceArrs(dct, self.totalN),
                                         in_axes=({'arr1':0, 'arr2':0},)))

        # transposeT = jnp.transpose(self.totalN)
        dct = {'arr1':self.SNew, 'arr2':self.amplitudes[0,:]}
        # self.slicedTotal = sliceArrsvmap(dct)

        self.slicedTotal = sliceArrs(dct, self.totalN)
        # self.slicedTotal = reduce(lambda x,y: jnp.vstack((x, jnp.where(y, transposeT, -1))), masks)

        self.slicedTotal = jnp.transpose(self.slicedTotal)
        self.totalNNotNorm = self.slicedTotal * self.WohlerC.NMax

        self.Nsamples = jax.vmap(jnp.exp, in_axes=(0,))(self.totalNNotNorm)

        print("slicedTotal shape-->",self.slicedTotal.shape)
        print("Nsamples shape -->",self.Nsamples.shape)

    #@profile
    def joinAmplitudesStress(self, maxLoads):

        print("Join amplitude and stress...")
        loads = list(self.sample_loads.values())[0]
        vHisto = jax.vmap(lambda x: jnp.histogram(x, bins=maxLoads), in_axes=(1,))
        self.cycles, self.amplitudes = vHisto(loads)

        # self.cycles, self.amplitudes = jnp.histogram(loads.flatten(), bins=maxLoads)
        # Transform to amplitudes and then to 0,1 in Wohler Space

        self.amplitudes /= self.WohlerC.SMax

        # self.cycles = self.cycles[:,0]
        # self.amplitudes = self.amplitudes[:,0]

        # vFillArray = jax.jit(jax.vmap(fillArray, in_axes=(0,)))
        # self.SNew = vFillArray(self.amplitudes)
        # vUnique = jax.vmap(lambda x: jnp.unique(x, size=180), in_axes=(0,))

        self.SNew = completeArray(self.amplitudes[0,:], num=30)
        print(f'SNew shape: {self.SNew.shape}')

    def sample_model(self, model:str, ndraws):
        if model == 'wohler':
            self.WohlerC.sampleModel(ndraws)
            self.WohlerC.samplePosterior()
            self.WohlerC.plotGP()

        elif model == 'loads':
            self.LoadM.sampleModel(ndraws)

    def plotLoadSamples(self):
        print("Plotting Samples...")
        loads = list(self.sample_loads.values())[0]
        loads = jnp.array(loads)
        vHisto = jax.vmap(lambda x: jnp.histogram(x), in_axes=(1,))
        counts, bins = vHisto(loads)
        _,ax = plt.subplots(1,1, figsize=(12,8))
        for i, (count, bin_) in enumerate(zip(counts, bins)):
            ax.hist(bin_[:-1], bin_, weights=count, density=True)
            if i>200:
                break
        plt.savefig(os.path.join(self.wohlerPath, 'loadSample2.jpg'))
        plt.close()

    def plotFatigueLifeSamples(self):
        newMean = jax.vmap(log1exp, in_axes=(0,))(self.meanSamples)
        newVar = jax.vmap(jnp.exp, in_axes=(0,))(self.varianceSamples)
        _,ax = plt.subplots(4,1, figsize=(20,14))

        plot_gp_dist(ax[0], self.totalN, self.SNew, palette="autumn")
        plot_gp_dist(ax[1], newMean, self.SNew, palette="autumn")
        plot_gp_dist(ax[2], newVar, self.SNew)
        plot_gp_dist(ax[3], self.varianceSamples, self.SNew)

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
