import os
import math
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.style.use(['science', 'ieee'])
from src.combinedModel import DamageCalculation

# Re sample posterior to plot properly
# @profile
def main():
    props = {
            'Eal':69e3,
            'Ecore':207e3,
            'rw':2.5,'rc':1,
            'layers':[26,7],
            'T':30641*0.15
                 }

    damageCal = DamageCalculation(wohlerPath='Results1',
                              loadObserved='data/800369.csv',
                              WohlerObserved='data/SN_curve.mat',
                              loadPath='Results2', cableProps=props)

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
    # scaleFactors = np.fromiter(map(lambda x: int(10**x), range(6)), dtype=np.int64)
    scaleFactors = np.arange(10, 10000, 100)
    p_failures = []
    var_coeff = []

    for i, scale in enumerate(scaleFactors):
        print('='*30)
        damages = damageCal.calculateDamage(scaleFactor=scale, _iter=i)
        # damages = damageCal.calculateDamage_debug(scaleFactor=scale, _iter=i)

        if isinstance(damages, jnp.DeviceArray):
            indicator = damages[damages>1]
            p_failures.append(len(indicator)/len(damages))
            N_mcs = len(damages)
            if np.isclose(p_failures[-1], 0):
                var_coeff.append(0)
            else:
                var_coeff.append(math.sqrt((1-p_failures[-1])/(N_mcs*p_failures[-1])))
            print(f'Probability of failure: {str(p_failures[-1])}')
            print(f'Variation Coeff: {str(var_coeff[-1])}')
        else:

            _,ax = plt.subplots(figsize=(12,8))
            ax.plot(ndraws*scaleFactors[:len(p_failures)], p_failures, label=f'Tension:25%RTS')
            ax.plot(ndraws*scaleFactors[:len(var_coeff)], var_coeff)
            ax.set_xlabel('Cycles Observed')
            ax.set_ylabel('Probability of Failure')
            ax.set_xscale('log')
            # ax.set_yscale('log')
            plt.savefig(os.path.join('Results1', 'PFailure.jpg'))
            return

if __name__=='__main__':
    main()
