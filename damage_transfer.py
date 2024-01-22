import os
import pathlib
import pymc as pm
import nutpie
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

BASE_PATH = pathlib.Path(__file__).parent
DATA_DIR = BASE_PATH / "data"
RESULTS_FOLDER = BASE_PATH / "RESULTS"
PLOT_DIR = BASE_PATH / "plots"

CYCLING_HOURS = 100
N_YEARS = 15
LOAD_PATH = RESULTS_FOLDER / "LOADS"
MAX_SAMPLES = 100000
year, i = 0, 13
nbatches = 100


def get_tot_damages(year, nbatches=100) -> np.ndarray:
    tot_damages = None
    for i in range(nbatches):
        print(f"BATCH NUMBER : {i}")

        try:
            with open(
                LOAD_PATH / f"tot_damages_year{year}_batch_{i}.npz", "rb"
            ) as file:
                samples = np.load(file, allow_pickle=True)
                # damages_aeran = {key: jnp.array(val) for key, val in samples.items()}
                # damages_aeran = jnp.array(samples['arr_0'])
                damages_aeran = samples["arr_0"]
                if tot_damages is not None:
                    tot_damages = np.hstack([tot_damages, damages_aeran])
                else:
                    tot_damages = damages_aeran
        except Exception as e:
            print(e)

    if tot_damages:
        return tot_damages[~np.isnan(tot_damages)][:MAX_SAMPLES]

    return tot_damages


tot_damages = get_tot_damages(0)

print("NAN FRAC ==>")
print(f"NAN LEN {len(tot_damages[np.isnan(tot_damages)])}")
print(len(tot_damages[np.isnan(tot_damages)]) / len(tot_damages))
print(f"TOTAL LEN {len(tot_damages)}")

tot_damages = np.array([get_tot_damages(i) for i in range(N_YEARS)], dtype=np.float32)

with pm.Model() as damage_model:
    alpha = pm.Gamma("alpha", alpha=1, beta=1, shape=(N_YEARS,))
    beta = pm.Gamma("beta", alpha=1, beta=1, shape=(N_YEARS,))
    damages = pm.Gamma("likelihood", alpha=alpha, beta=beta, observed=tot_damages)


compiled_model = nutpie.compile_pymc_model(damage_model)
trace = nutpie.sample(compiled_model, draws=1400, tune=1000, chains=4)
az.to_netcdf(data=trace, filename=RESULTS_FOLDER / "DAMAGE_MODEL_TRACE.nc")
with damage_model:
    ppc = pm.sample_posterior_predictive(
        trace,
    )

# fig, ax = plt.subplots(1,1)
az.plot_trace(trace)
plt.savefig(RESULTS_FOLDER / "damage_model_posterior.png", dpi=600)
plt.close()

# fig, ax = plt.subplots(1,1)
# az.plot_ppc(ppc, ax=ax)
az.plot_ppc(ppc)
plt.savefig(RESULTS_FOLDER / "damage_model_ppc.png", dpi=600)
plt.close()
