from src.combinedModel import DamageCalculation

# Re sample posterior to plot properly

def main():

    damageCal = DamageCalculation('Results1',
                              loadObserved='data/800369.csv',
                              WohlerObserved='data/SN_curve.mat',
                              loadPath='Results2')
    # damageCal.sample_model('wohler', 2000)
    # damageCal.sample_model('loads', 2000)
    # damageCal.WohlerC.samplePosterior()
    # damageCal.WohlerC.plotGP()
    damageCal.restoreLoadSamples(ndraws=20000)
    damageCal.restoreFatigueLifeSamples(maxLoads=20)
    damageCal.plotFatigueLifeSamples()
    damageCal.plotLoadSamples()
    # damages = damageCal.calculateDamage_debug()

    print('max vals WohlerC-->')
    print(damageCal.WohlerC.NMax)
    print(damageCal.WohlerC.SMax)

    print('max vals Loads-->')
    print(damageCal.LoadM.maxAmp)

    damages = damageCal.calculateDamage()
    print(damages)

if __name__=='__main__':
    main()
