from src.models import DamageCalculation

damage = DamageCalculation('Results1', 'data/SN_curve.mat')
damage.build_HeteroskedasticModel()
# damage.sampleModel(2000)
# damage.samplePosterior()
damage.plotGP()
