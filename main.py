from src.combinedModel import DamageCalculation

damageCal = DamageCalculation('Results1',
                              loadObserved='data/800369.csv',
                              WohlerObserved='data/SN_curve.mat',
                              loadPath='Results2')
damageCal.restoreLoadSamples(ndraws=20000)
# Re sample posterior to plot properly
# damageCal.sample_model('wohler', 3000)
damageCal.WohlerC.samplePosterior()
damageCal.WohlerC.plotGP()
damageCal.restoreFatigueLifeSamples(maxLoads=50)
damageCal.plotFatigueLifeSamples()
damages = damageCal.calculateDamage_debug()
