import os
import pathlib
import pymc as pm
import nutpie
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

BASE_PATH = pathlib.Path(__file__).parent
DATA_DIR = BASE_PATH / "data"
RESULTS_FOLDER = BASE_PATH / "RESULTS"
PLOT_DIR = BASE_PATH / "plots"

CYCLING_HOURS = 100
N_YEARS = 5
LOAD_PATH = RESULTS_FOLDER / "LOADS"
MAX_SAMPLES = 50000
nbatches = 100


def get_tot_damages(year, nbatches=100) -> np.ndarray:
    tot_damages = None
    for i in range(nbatches):
        # print(f"BATCH NUMBER : {i}")

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
            continue
            # print(e)

    if tot_damages is not None:
        print("NAN FRAC ==>")
        print(f"NAN LEN {len(tot_damages[np.isnan(tot_damages)])}")
        print(len(tot_damages[np.isnan(tot_damages)]) / len(tot_damages))
        print(f"TOTAL LEN {len(tot_damages)}")
        return tot_damages[~np.isnan(tot_damages)][:MAX_SAMPLES]

    return tot_damages


damages = [get_tot_damages(i) for i in range(N_YEARS)]
tot_damages = np.array([d for d in damages if d is not None], dtype=np.float32)
coords = {
    "N_YEARS": np.arange(tot_damages.shape[0]),
    "obs": np.arange(tot_damages.shape[1]),
}

# with pm.Model(coords=coords) as damage_model:
with pm.Model() as damage_model:
    for i in coords["N_YEARS"]:
        alpha = pm.Gamma(f"alpha_{i}", alpha=1, beta=1)
        beta = pm.Gamma(f"beta_{i}", alpha=1, beta=1)
        damages = pm.Gamma(
            f"damage_{i}",
            alpha=alpha,
            beta=beta,
            observed=tot_damages[i, :],
            # dims=("N_YEARS", "obs"),
        )

if not os.path.exists(RESULTS_FOLDER / f"DAMAGE_MODEL_TRACE.nc"):
    compiled_model = nutpie.compile_pymc_model(damage_model)
    trace = nutpie.sample(compiled_model, draws=1400, tune=1000, chains=4)
    az.to_netcdf(data=trace, filename=RESULTS_FOLDER / "DAMAGE_MODEL_TRACE.nc")
else:
    trace = az.from_netcdf(RESULTS_FOLDER / f"DAMAGE_MODEL_TRACE.nc")

names = [f"damage_{i}-{i-1}" for i in range(1, N_YEARS)]
if not os.path.exists(RESULTS_FOLDER / f"damage_posterior.nc"):
    with damage_model:
        for j, i in enumerate(range(1, N_YEARS + 1)):
            if i == 1:
                damage_prev = damage_model.named_vars[f"damage_{i-1}"]
            else:
                damage_prev = damage_model.named_vars[names[j - 1]]
            # name = f"damage_{i}-{i-1}"
            # names.append(name)
            damage = damage_model.named_vars.get(f"damage_{i}")
            d = pm.Deterministic(names[j], damage + damage_prev)
        ppc = pm.sample_posterior_predictive(trace, var_names=names)
        az.to_netcdf(ppc, RESULTS_FOLDER / f"damage_posterior.nc")
else:
    ppc = az.from_netcdf(RESULTS_FOLDER / f"damage_posterior.nc")

fig, ax = plt.subplots(len(names))
fig.set_size_inches(3.1, 6.3)
plt.subplots_adjust(wspace=0.03175)
for i, n in enumerate(names):
    d = ppc.posterior_predictive[n]
    az.plot_dist(d, color="C1", label=n, ax=ax[i])

plt.savefig(RESULTS_FOLDER / "partial_damage.png", dpi=600)
plt.close()

# fig, ax = plt.subplots(1,1)
az.plot_trace(trace)
plt.savefig(RESULTS_FOLDER / "damage_model_posterior.png", dpi=600)
plt.close()

# fig, ax = plt.subplots(1,1)
# az.plot_ppc(ppc, ax=ax)
# az.plot_ppc(ppc)
# plt.savefig(RESULTS_FOLDER / "damage_model_ppc.png", dpi=600)
# plt.close()
