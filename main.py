#!/home/blueman69/miniforge3/envs/fem_gp/bin/python

import os
import pathlib
import pprint
import math
import numpy as np
import pytensor
import arviz as az
import jax.numpy as jnp
from jax import Array, devices
import matplotlib.pyplot as plt
# plt.style.use(['science', 'ieee'])
from src.combinedModel import DamageCalculation

print(f"Running on: {devices()}")

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
#     jax.device_count()
# )
BASE_PATH = pathlib.Path(__file__).parent
DATA_DIR = BASE_PATH / "data"
RESULTS_FOLDER = BASE_PATH / "RESULTS"
PLOT_DIR = BASE_PATH / "plots"
NDRAWS = 2000
N_HIDDEN = 100

floatX = pytensor.config.floatX
RANDOM_SEED = 9927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

printer = pprint.PrettyPrinter(5, compact=True)

def getPFailure(damages:Array):
    return len(damages[damages>1])/len(damages)

def getVarCoeff(p_failures, N_mcs):
    return math.sqrt((1-p_failures)/(N_mcs*p_failures))


os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".75"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# Re sample posterior to plot properly
# @profile
def main(T):

    # scaleFactors = np.fromiter(map(lambda x: int(10**x), range(6)), dtype=np.int64)
    scaleFactors = np.arange(10, 10000, 100)

    print('-x'*30)
    print(f'T:{T}')
    props = {
            'Eal':69e3,
            'Ecore':207e3,
            'rw':2.5,'rc':1,
            'layers':[26,7],
            # 'T':30641*0.15
            'T':97.4*1000*T/100
                 }

    damageCal = DamageCalculation(wohlerPath=RESULTS_FOLDER / 'WOHLER',
                              loadObserved=DATA_DIR / '800369.csv',
                              WohlerObserved=DATA_DIR / 'SN_curve.mat',
                              loadPath=RESULTS_FOLDER / 'LOADS', cableProps=props,
                              PFObserved=DATA_DIR / 'pfData.csv',
                              Tpercentage=T)

    print('max vals WohlerC-->')
    print('logN-->', damageCal.WohlerC.NMax)
    print('SMax-->', damageCal.WohlerC.SMax)
    print('max vals Loads-->', damageCal.LoadM.maxAmp)
    # damageCal.sample_model('wohler', 2000)
    # damageCal.sample_model('loads', 2000)

    ndraws = 45000
    damageCal.restoreLoadSamples(ndraws=ndraws)
    damageCal.restoreFatigueLifeSamples(maxLoads=30)
    damageCal.plotFatigueLifeSamples()

    # damageCal.plotLoadSamples()
    # damages = damageCal.calculateDamage_debug()
    print('Calculating Damage...')
    vals = {'Miner':{'pFailures':[], 'varCoeff':[]}, 
            'Gao':{'pFailures':[], 'varCoeff':[]}}

    for i, scale in enumerate(scaleFactors):
        print('='*30)
        damagesGao = damageCal.calculateDamage(scaleFactor=scale, _iter=i)
        damagesMiner = damageCal.calculateDamageMiner(scaleFactor=scale, _iter=i)
        # damages = damageCal.calculateDamage_debug(scaleFactor=scale, _iter=i)

        if isinstance(damagesGao, Array):
            # indicator = damagesGao[damagesGao>1]

            vals['Gao']['pFailures'].append(getPFailure(damagesGao))
            vals['Miner']['pFailures'].append(getPFailure(damagesMiner))

            N_mcs = len(damagesGao)
            if np.isclose(vals['Gao']['pFailures'][-1], 0) or np.isclose(vals['Miner']['pFailures'][-1], 0):
                vals['Gao']['varCoeff'].append(0)
                vals['Miner']['varCoeff'].append(0)
            else:
                vals['Miner']['varCoeff'].append(getVarCoeff(vals['Miner']['pFailures'][-1], N_mcs))
                vals['Gao']['varCoeff'].append(getVarCoeff(vals['Gao']['pFailures'][-1], N_mcs))

            printer.pprint(vals)
            # print(f'Probability of failure: {str(p_failures[-1])}')
            # print(f'Variation Coeff: {str(var_coeff[-1])}')
        else:
            return vals, scaleFactors, ndraws

if __name__=='__main__':
    T = 20
    _,ax = plt.subplots(figsize=(12,8))

    vals, scaleFactors, ndraws = main(T)
    for model, values in vals.items():
        p_failures = values['pFailures']
        var_coeff = values['varCoeff']
        ax.plot(ndraws*scaleFactors[:len(p_failures)], p_failures, label=f'{model} Model at Tension:{T}%RTS')
        # ax.plot(ndraws*scaleFactors[:len(var_coeff)], var_coeff)
    ax.set_xlabel('Cycles Observed')
    ax.set_ylabel('Probability of Failure')
    ax.set_xscale('log')
    ax.legend()
        # ax.set_yscale('log')
        
plt.savefig(os.path.join('Results1', 'PFailure.jpg'))
plt.close()


