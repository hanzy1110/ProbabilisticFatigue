import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import jax.numpy as jnp
import jax
import json

from .damageModelGao import gaoModel
from .freqModel import LoadModel
from .models import WohlerCurve


class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self,obj)

def amplitudesToLoads(amplitudes, pf_slope):
    return amplitudes*pf_slope

def log1exp(arr:jnp.ndarray):
    return jnp.log(1+jnp.exp(arr))

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

    def restoreTraces(self):
            self.traceWohler = self.WohlerC.restoreTrace()
            self.traceLoads = self.LoadM.restoreTrace()

    def sampleLoads(self):
        with self.LoadM.amplitude_model:
            pf_slope = pm.Normal('pf_slope', mu=5, sigma=0.5)
            loads = pm.Deterministic('Loads', amplitudesToLoads(self.LoadM.loads, pf_slope))
            self.sample_loads = pm.sample_posterior_predictive(trace=self.traceLoads, var_names=['Loads'])

    def transformGPs(self):

        self.meanSamples = jnp.array(self.samples['meanN'])
        self.varianceSamples = jnp.array(self.samples['sigmaN'])
        self.totalN = jax.vmap(log1exp, in_axes=(0,))(self.meanSamples )+ jax.vmap(jnp.exp, in_axes=(0,))(self.varianceSamples)

        self.totalNNotNorm = self.totalN * self.WohlerC.NMax

        self.Nsamples = jax.vmap(jnp.exp, in_axes=(0,))(self.totalNNotNorm)

    def sampleFatigueLife(self, maxLoads:int):
        self.cycles, amplitudes = jnp.histogram(self.sample_loads['Loads'].flatten(), bins=maxLoads)

        # Transform to amplitudes and then to 0,1 in Wohler Space
        amplitudes = amplitudes* self.LoadM.maxAmp/self.WohlerC.SMax

        with self.WohlerC.SNCurveModel:
            meanN = self.WohlerC.gp_ht.conditional('meanN', Xnew=amplitudes.reshape((-1,1)))
            sigmaN = self.WohlerC.σ_gp.conditional('sigmaN', Xnew=amplitudes.reshape((-1,1)))
            # totalN = pm.Deterministic('totalN', meanN + sigmaN)
            self.samples = pm.sample_posterior_predictive(self.traceWohler, var_names=['meanN', 'sigmaN'])

        with open(os.path.join(self.wohlerPath, 'lifeSamples.json'), 'w') as file:
            json.dump(self.samples, file, cls=npEncoder)

        self.transformGPs()

    def restoreFatigueLifeSamples(self, maxLoads):

        if os.path.exists(os.path.join(self.wohlerPath, 'lifeSamples.json')):
            self.cycles, amplitudes = jnp.histogram(self.sample_loads['Loads'], bins=maxLoads)
            with open(os.path.join(self.wohlerPath, 'lifeSamples.json'), 'r') as file:
                samples = json.load(file)
            self.samples = {key:jnp.array(val) for key, val in samples.items()}
            self.transformGPs()
        else:
            self.sampleFatigueLife(maxLoads=maxLoads)

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
