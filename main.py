from src.combinedModel import DamageCalculation

# Re sample posterior to plot properly
def main():
    props = {
            'Eal':69e3,
            'Ecore':200e3,
            'rw':2.5,'rc':1,
            'layers':[26,7],
            'T':30641*0.05
                 }

    damageCal = DamageCalculation(wohlerPath='Results1',
                              loadObserved='data/800369.csv',
                              WohlerObserved='data/SN_curve.mat',
                              loadPath='Results2', cableProps=props)

    print('max vals WohlerC-->')
    print('logN', damageCal.WohlerC.NMax)
    print('SMax', damageCal.WohlerC.SMax)
    # damageCal.sample_model('wohler', 2000)

    # damageCal.sample_model('loads', 2000)
    # damageCal.WohlerC.samplePosterior()
    # damageCal.WohlerC.plotGP()
    damageCal.restoreLoadSamples(ndraws=2000)
    damageCal.restoreFatigueLifeSamples(maxLoads=20)
    # damageCal.plotFatigueLifeSamples()
    # damageCal.plotLoadSamples()
    damages = damageCal.calculateDamage_debug()


    print('max vals Loads-->')
    print(damageCal.LoadM.maxAmp)

    print(damages)

if __name__=='__main__':
    main()
