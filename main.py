from src.combinedModel import DamageCalculation

# Re sample posterior to plot properly
def main():
    props = {
            'Eal':70e3,
            'Ecore':210e3,
            'rw':2.5,'rc':1,
            'layers':[26,7],
            'T':30641*0.00005
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
    # damageCal.WohlerC.samplePosterior()
    # damageCal.WohlerC.plotGP()
    damageCal.restoreLoadSamples(ndraws=20000)
    damageCal.restoreFatigueLifeSamples(maxLoads=20)
    # damageCal.plotFatigueLifeSamples()
    # damageCal.plotLoadSamples()
    # damages = damageCal.calculateDamage_debug()
    print('Calculating Damage...')
    damages = damageCal.calculateDamage()
    print(damages)

if __name__=='__main__':
    main()
