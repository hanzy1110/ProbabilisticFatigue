from src.models import DamageCalculation
from src.freqModel import FatigueModel

fatigue = FatigueModel(resultsFolder='Results2', observedDataPath='data/800356.csv', b_mean=-0.1)
fatigue.buildMixtureFreqModel()
fatigue.sampleModel(3000)
fatigue.samplePosterior()


# damage = DamageCalculation('Results1', 'data/SN_curve.mat')
# damage.build_HeteroskedasticModel()
# damage.sampleModel(2000)
# damage.samplePosterior()
# damage.plotGP()
