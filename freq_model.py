# Maybe later compute a frequency distribution
import pymc as pm
import pytensor.tensor as pt
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

from src.freqModel import calculate_freq_dist


def stick_breaking(beta):
    portion_remaining = pt.concatenate([[1], pt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining


BASE_PATH = pathlib.Path(__file__).parent
DATA_DIR = BASE_PATH / "data"
RANDOM_SEED = 123
freqs = calculate_freq_dist(DATA_DIR / "800369.csv")

freq_hist, freq_bins = np.histogram(freqs)
N = freq_hist.shape[0]
K = 30

fig, ax = plt.subplots(1, 1)

ax.hist(freqs, density=True)
ax.set_xlabel("Frecuencia")
ax.set_ylabel("Densidad")
plt.savefig(BASE_PATH / "RESULTS/freq_distro.png", dpi=600)

if False:
    with pm.Model(coords={"component": np.arange(K), "obs_id": np.arange(N)}) as model:
        alpha = pm.Gamma("alpha", 1.0, 1.0)
        beta = pm.Beta("beta", 1, alpha, dims="component")
        w = pm.Deterministic("w", stick_breaking(beta), dims="component")
        # Gamma is conjugate prior to Poisson
        lambda_ = pm.Gamma("lambda_", 300.0, 2.0, dims="component")
        obs = pm.Mixture(
            "obs",
            w,
            pm.Poisson.dist(lambda_),
            observed=freq_hist,
            dims="obs_id",
        )

    with model:
        trace = pm.sample(
            tune=2500,
            init="advi",
            target_accept=0.975,
            random_seed=RANDOM_SEED,
        )
