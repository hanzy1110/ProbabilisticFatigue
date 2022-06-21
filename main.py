from src.combinedModel import DamageCalculation

# Re sample posterior to plot properly
# damageCal.sample_model('wohler', 3000)
# damageCal.WohlerC.samplePosterior()
# damageCal.WohlerC.plotGP()

def main():

    damageCal = DamageCalculation('Results1',
                              loadObserved='data/800369.csv',
                              WohlerObserved='data/SN_curve.mat',
                              loadPath='Results2')
    damageCal.restoreLoadSamples(ndraws=50000)
    damageCal.restoreFatigueLifeSamples(maxLoads=50)
    damageCal.plotFatigueLifeSamples()
    damages = damageCal.calculateDamage_debug()

if __name__=='__main__':
    main()
