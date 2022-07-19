from src.combinedModel import DamageCalculation
import math

# Re sample posterior to plot properly
# @profile
def main():
    props = {
            'Eal':69e3,
            'Ecore':207e3,
            'rw':2.5,'rc':1,
            'layers':[26,7],
            'T':30641*0.55
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

    damageCal.restoreLoadSamples(ndraws=20000)
    damageCal.restoreFatigueLifeSamples(maxLoads=30)
    damageCal.plotFatigueLifeSamples()

    # damageCal.plotLoadSamples()
    # damages = damageCal.calculateDamage_debug()
    print('Calculating Damage...')
    damages = damageCal.calculateDamage(scaleFactor=500)

    indicator = len(damages[damages>1])
    p_failure = len(indicator)/len(damages)
    N_mcs = len(damages)
    var_coeff = math.sqrt((1-p_failure)/(N_mcs*p_failure))
    print(f'Probability of failure: {str(p_failure)[:5]}')
    print(f'Variation Coeff: {str(var_coeff)[:5]}')
    # damages = damageCal.calculateDamageMiner()
    print(damages)

if __name__=='__main__':
    main()
