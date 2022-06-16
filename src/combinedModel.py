import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import jax.numpy as jnp
import jax

from damageModelGao import gaoModel
from freqModel import LoadModel
from models import WohlerCurve

def amplitudesToLoads(amplitudes, pf_slope):
    return amplitudes*pf_slope

class DamageCalculation:
    def __init__(self, wohlerPath:str, loadPath:str,
                 WohlerObserved:str, loadObserved:str,
                 restoreTrace:bool=True)->None:

        self.WohlerC = WohlerCurve(resultsFolder=wohlerPath,
                                   observedDataPath=WohlerObserved)

        self.LoadM = LoadModel(resultsFolder=loadPath,
                               observedDataPath=loadObserved,
                               b_mean=-0.1)

        self.WohlerC.build_HeteroskedasticModel()
        self.LoadM.buildMixtureFreqModel()

        if restoreTrace:
            self.restoreTraces()

    def restoreTraces(self):
            self.traceWohler = self.WohlerC.restoreTrace()
            self.traceLoads = self.LoadM.restoreTrace()

    def sampleLoads(self):
        with self.LoadM.frequency_model:
            pf_slope = pm.Normal('pf_slope', mu=1000, sigma=100)
            loads = pm.Deterministic('Loads', amplitudesToLoads(self.LoadM.loads, pf_slope))
            self.sample_loads = pm.sample_posterior_predictive(trace=self.traceLoads, var_names=['Loads'])

    def sampleFatigueLife(self):
        self.cycles, amplitudes = jnp.histogram(self.sample_loads['Loads'])

        with self.WohlerC.SNCurveModel:
            meanN = self.WohlerC.gp_ht.conditional('meanN', Xnew=amplitudes.reshape((-1,1)))
            sigmaN = self.WohlerC.σ_gp.conditional('sigmaN', Xnew=amplitudes.reshape((-1,1)))
            totalN = pm.Deterministic('totalN', meanN + sigmaN)
            self.logNsamples = pm.sample_posterior_predictive(self.traceWohler, var_names=['totalN'])

        self.Nsamples = jnp.exp(logNsamples)

    def calculateDamage(self):
        damages = []
        for Nsample in self.Nsamples:
            damages.append(gaoModel(self.cycles, Nsample))

        return damages
